import io
import os
import sys
import tempfile
import threading
import uuid
from flask import Flask, request, jsonify, render_template, send_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB limit

# In-memory job store
jobs = {}
jobs_lock = threading.Lock()


def _run_job(job_id, source_path, target_path, options):
    """Run the matching in a background thread and update job state."""
    from match_core import run_matching_core

    with jobs_lock:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["log"] = []

    log_lines = []

    def emit(msg):
        with jobs_lock:
            jobs[job_id]["log"].append(msg)

    try:
        output_path = tempfile.mktemp(suffix=".csv")
        summary = run_matching_core(
            source_path=source_path,
            target_path=target_path,
            output_path=output_path,
            log_fn=emit,
            **options,
        )
        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["output_path"] = output_path
            jobs[job_id]["summary"] = summary
    except Exception as exc:
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(exc)
    finally:
        # Clean up uploaded temp files
        try:
            os.unlink(source_path)
        except Exception:
            pass
        try:
            os.unlink(target_path)
        except Exception:
            pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    source_file = request.files.get("source")
    target_file = request.files.get("target")

    if not source_file or not target_file:
        return jsonify({"error": "Both source and target CSV files are required."}), 400

    # Save uploads to temp files
    src_tmp = tempfile.mktemp(suffix=".csv")
    tgt_tmp = tempfile.mktemp(suffix=".csv")
    source_file.save(src_tmp)
    target_file.save(tgt_tmp)

    # Parse options from form
    def fv(key, default):
        v = request.form.get(key, "").strip()
        return v if v else default

    options = {
        "source_col": fv("source_col", "company"),
        "target_col": fv("target_col", "company"),
        "source_id_col": fv("source_id_col", ""),
        "target_id_col": fv("target_id_col", "account_id"),
        "model": fv("model", "text-embedding-3-small"),
        "threshold": float(fv("threshold", "0.82")),
        "topk": int(fv("topk", "3")),
        "batch_size": int(fv("batch_size", "128")),
        "api_key": fv("api_key", ""),
    }

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "log": [], "output_path": None, "summary": None, "error": None}

    t = threading.Thread(target=_run_job, args=(job_id, src_tmp, tgt_tmp, options), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status": job["status"],
        "log": job["log"],
        "summary": job["summary"],
        "error": job["error"],
    })


@app.route("/api/download/<job_id>")
def api_download(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    output_path = job["output_path"]
    if not output_path or not os.path.exists(output_path):
        return jsonify({"error": "Output file missing"}), 404
    return send_file(
        output_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name="matches.csv",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
