import uvicorn
from fastapi import FastAPI, Body, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json
import io
import base64
from typing import List, Union
from threading import Lock
import modules.shared as shared
import modules.sd_models

generate_lock = Lock()

class TextToImageModelOpts(BaseModel):
    checkpoint: str = Field(default="", title="Inference Model Checkpoints")
    hypernet: str = Field(default="None", title="Hypernetwork to use for finetune")

class TextToImage(BaseModel):
    prompt: str = Field(default="", title="Prompt Text", description="The text to generate an image from.")
    negative_prompt: str = Field(default="", title="Negative Prompt Text")
    prompt_style: str = Field(default="None", title="Prompt Style")
    prompt_style2: str = Field(default="None", title="Prompt Style 2")
    steps: int = Field(default=20, title="Steps")
    sampler_index: int = Field(0, title="Sampler Index")
    restore_faces: bool = Field(default=False, title="Restore Faces")
    tiling: bool = Field(default=False, title="Tiling")
    n_iter: int = Field(default=1, title="N Iter")
    batch_size: int = Field(default=1, title="Batch Size")
    cfg_scale: float = Field(default=7, title="Classifier-Free Guidance Scale (How strict the generation will adhere to the prompt)")
    seed: int = Field(default=-1.0, title="Seed")
    subseed: int = Field(default=-1.0, title="Subseed")
    subseed_strength: float = Field(default=0, title="Subseed Strength")
    seed_resize_from_h: int = Field(default=0, title="Seed Resize From Height")
    seed_resize_from_w: int = Field(default=0, title="Seed Resize From Width")
    seed_enable_extras: bool = Field(default=False, title="Enable Extra Seed Functions")
    height: int = Field(default=512, title="Height")
    width: int = Field(default=512, title="Width")
    enable_hr: bool = Field(default=False, title="Enable HR")
    scale_latent: bool = Field(default=True, title="Scale Latent")
    denoising_strength: float = Field(default=0.7, title="Denoising Strength")
    modelOptions: Union[TextToImageModelOpts, None] = Field(default=None, title="Options for SD model")

class TextToImageResponse(BaseModel):
    images: List[str] = Field(default=None, title="Image", description="The generated image in base64 format.")
    all_prompts: List[str] = Field(default=None, title="All Prompts", description="The prompt text.")
    negative_prompt: str = Field(default=None, title="Negative Prompt Text")
    seed: int = Field(default=None, title="Seed")
    all_seeds: List[int] = Field(default=None, title="All Seeds")
    subseed: int = Field(default=None, title="Subseed")
    all_subseeds: List[int] = Field(default=None, title="All Subseeds")
    subseed_strength: float = Field(default=None, title="Subseed Strength")
    width: int = Field(default=None, title="Width")
    height: int = Field(default=None, title="Height")
    sampler_index: int = Field(default=None, title="Sampler Index")
    sampler: str = Field(default=None, title="Sampler")
    cfg_scale: float = Field(default=None, title="Config Scale")
    steps: int = Field(default=None, title="Steps")
    batch_size: int = Field(default=None, title="Batch Size")
    restore_faces: bool = Field(default=None, title="Restore Faces")
    face_restoration_model: str = Field(default=None, title="Face Restoration Model")
    sd_model_hash: str = Field(default=None, title="SD Model Hash")
    seed_resize_from_w: int = Field(default=None, title="Seed Resize From Width")
    seed_resize_from_h: int = Field(default=None, title="Seed Resize From Height")
    denoising_strength: float = Field(default=None, title="Denoising Strength")
    extra_generation_params: dict = Field(default={}, title="Extra Generation Params")
    index_of_first_image: int = Field(default=None, title="Index of First Image")
    html: str = Field(default=None, title="HTML")

class JobStateResponse(BaseModel):
    skipped: bool = Field(default=None, title="Skipped Job")
    interrupted: bool = Field(default=None, title="Interrupted Job")
    job: str = Field(default=None, title="Job Name")
    job_no: int = Field(default=None, title="Job No.")
    job_count: int = Field(default=None, title="Job Count")
    job_timestamp:str = Field(default=None, title="Job Timestamp")
    sampling_step: int = Field(default=None, title="Current Job Sampling Step")
    sampling_steps: int = Field(default=None, title="Total Job Sampling Step")
    current_image_sampling_step: int = Field(default=None, title="Current Image Sampling Steps")
    model_loading: bool = Field(default=None, title="Model Loading")

class SdInfoResponse(BaseModel):
    checkpoints: list = Field(default=None, title="List of installed Checkpoints")
    hypernets: list = Field(default=None, title="List of hypernetwork to use")

app = FastAPI()


class Api:
    def __init__(self, txt2img, img2img, run_extras, run_pnginfo):
        self.txt2img = txt2img
        self.img2img = img2img
        self.run_extras = run_extras
        self.run_pnginfo = run_pnginfo

        self.router = APIRouter()
        app.add_api_route("/v1/txt2img", self.txt2imgendoint, response_model=TextToImageResponse, methods=['post'])
        # app.add_api_route("/v1/img2img", self.img2imgendoint)
        # app.add_api_route("/v1/extras", self.extrasendoint)
        # app.add_api_route("/v1/pnginfo", self.pnginfoendoint)
        app.add_api_route("/v1/state", self.getJobStateEndpoint, response_model=JobStateResponse, methods=['get'])
        app.add_api_route("/v1/op/interrupt", self.interruptJobEndpoint)
        app.add_api_route("/v1/op/skip", self.skipJobEndpoint)
        app.add_api_route("/v1/stable_diffusion",self.SdInfoEndpoint, response_model=SdInfoResponse)

    def txt2imgendoint(self, txt2imgreq: TextToImage = Body(embed=True)):
        # aquire lock, force single job only
        if generate_lock.locked():
            raise HTTPException(status_code=503, detail="Another Generation is in progress!")
        generate_lock.acquire()
        shared.state.reset_state()

        # load model checkpoint
        model_checkpoint = txt2imgreq.modelOptions.checkpoint
        checkpoint_info = modules.sd_models.checkpoints_list.get(model_checkpoint, None)
        if checkpoint_info == None:
            raise HTTPException(status_code=401, detail="Checkpoint does not exist")
        shared.sd_model = modules.sd_models.load_model(checkpoint_info)
        modules.sd_models.reload_model_weights(shared.sd_model, checkpoint_info)

        # hypernet
        shared.opts.__setattr__('sd_hypernetwork', txt2imgreq.modelOptions.hypernet)
        
        # process
        images, params, html = self.txt2img(
            prompt=txt2imgreq.prompt,
            negative_prompt=txt2imgreq.negative_prompt, 
            prompt_style=txt2imgreq.prompt_style,
            prompt_style2=txt2imgreq.prompt_style2,
            steps=txt2imgreq.steps,
            sampler_index=txt2imgreq.sampler_index,
            restore_faces=txt2imgreq.restore_faces,
            tiling=txt2imgreq.tiling,
            n_iter=txt2imgreq.n_iter,
            batch_size=txt2imgreq.batch_size,
            cfg_scale=txt2imgreq.cfg_scale,
            seed=txt2imgreq.seed,
            subseed=txt2imgreq.subseed,
            subseed_strength=txt2imgreq.subseed_strength,
            seed_resize_from_h=txt2imgreq.seed_resize_from_h,
            seed_resize_from_w=txt2imgreq.seed_resize_from_w,
            seed_enable_extras=txt2imgreq.seed_enable_extras,
            height=txt2imgreq.height,
            width=txt2imgreq.width,
            enable_hr=txt2imgreq.enable_hr,
            scale_latent=txt2imgreq.scale_latent,
            denoising_strength=txt2imgreq.denoising_strength
        )
        b64images = []
        for i in images:
            buffer = io.BytesIO()
            i.save(buffer, format="png")
            b64images.append(base64.b64encode(buffer.getvalue()))
        resp_params = json.loads(params)
        generate_lock.release()
        return TextToImageResponse(images=b64images, **resp_params, html=html)

    def img2imgendoint(self):
        raise NotImplementedError

    def extrasendoint(self):
        raise NotImplementedError

    def pnginfoendoint(self):
        raise NotImplementedError

    def getJobStateEndpoint(self):
        return JobStateResponse(
            skipped=shared.state.skipped,
            interrupted=shared.state.interrupted,
            job=shared.state.job,
            job_no=shared.state.job_no,
            job_count=shared.state.job_count,
            job_timestamp=shared.state.job_timestamp,
            sampling_step=shared.state.sampling_step,
            sampling_steps=shared.state.sampling_steps,
            current_image_sampling_step=shared.state.current_image_sampling_step,
            model_loading=shared.state.is_model_loading
        )

    def interruptJobEndpoint(self):
        shared.state.interrupt()
        return "irq ok"
    
    def skipJobEndpoint(self):
        shared.state.skip()
        return "skip ok"

    def SdInfoEndpoint(self):
        return SdInfoResponse(
            checkpoints=[key._asdict() for key in modules.sd_models.checkpoints_list.values()],
            hypernets= [x for x in shared.hypernetworks.keys()]
        )

    def launch(self, server_name, port):
        app.include_router(self.router)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if shared.cmd_opts.allow_api_cors else [],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        uvicorn.run(app, host=server_name, port=port)
