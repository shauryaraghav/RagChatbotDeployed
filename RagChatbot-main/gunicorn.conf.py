import os

workers = 1
worker_class = 'uvicorn.workers.UvicornWorker'
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"
timeout = 120
