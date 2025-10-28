from pathlib import Path

__all__ = ["run_demo"]


def run_demo(demo, static_dir=None, port=6150):
    import gradio as gr
    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    app = FastAPI()
    if static_dir:
        static_dir = Path(static_dir).absolute()
        app.mount(
            "/static",
            StaticFiles(directory=static_dir, html=True),
            name="static",
        )
    app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host="0.0.0.0", port=port)
