from pathlib import Path

import gradio as gr
import torch
import uuid
from drecg.models.feat_extraction import VitLaionFeatureExtractor
import shutil
from queue import Queue, Full
from threading import Thread


class HFPetDatasetManager(Thread):
    def __init__(self, ds_name, hf_token, queue, local_path='collected'):
        Thread.__init__(self)
        self.queue = queue
        import huggingface_hub
        repo_id = huggingface_hub.get_full_repo_name(
            ds_name, token=hf_token
        )
        self.path_to_dataset_repo = huggingface_hub.create_repo(
            repo_id=repo_id,
            token=hf_token,
            private=True,
            repo_type="dataset",
            exist_ok=True,
        )
        self.repo = huggingface_hub.Repository(
            local_dir=local_path,
            clone_from=self.path_to_dataset_repo,
            use_auth_token=hf_token,
        )
        self.repo.git_pull()
        self.mistakes_dir = Path(local_path) / "mistakes"
        self.normal_dir = Path(local_path) / "normal"

        self.true_different_dir = self.normal_dir / "different"
        self.true_same_dir = self.normal_dir / "same"

        self.false_different_dir = self.mistakes_dir / "different"
        self.false_same_dir = self.mistakes_dir / "same"

        self.true_same_dir.mkdir(parents=True, exist_ok=True)
        self.true_different_dir.mkdir(parents=True, exist_ok=True)
        self.false_same_dir.mkdir(parents=True, exist_ok=True)
        self.false_different_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        while True:
            _signal = self.queue.get()
            self.repo.git_pull()
            self.repo.push_to_hub(commit_message=f"Upload data changes...")
            print('Changes pushed to dataset!')


ds_manager_queue = Queue(maxsize=1)
model_cls = None
feat_extractor = None
processor = None
HF_API_TOKEN = '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'
device = torch.device("cuda")
dataset_name = "pet-cls-mistakes-2"
ds_manager = None


def push_files_async():
    try:
        ds_manager_queue.put_nowait('Ok')
        print('DS upload requested!')
    except Full:
        print('Pull already started!')


def predict_diff(img_a, img_b):
    global model_cls, feat_extractor, processor
    x = processor(img_a).unsqueeze(dim=0).to(device), processor(img_b).unsqueeze(dim=0).to(device)
    a, b = feat_extractor(x)
    proba = torch.sigmoid(model_cls(a, b)).item()
    score_str = "{:.2f}".format(round(proba) * proba + round(1 - proba) * (1 - proba))
    base_name = f"{str(uuid.uuid4()).replace('-', '')}-{score_str}"
    save_image_pairs(img_a, img_b, proba, base_name)
    return {'Same': proba, 'Different': 1 - proba}, base_name


def save_image_pairs(img_a, img_b, proba, base_name):
    sub_dir = 'same' if proba > 0.5 else 'different'
    img_a.save(f'collected/normal/{sub_dir}/{base_name}_a.png')
    img_b.save(f'collected/normal/{sub_dir}/{base_name}_b.png')
    push_files_async()


def move_to_flagged(base_name: str, label: str):
    sub_dir = label.lower()
    destination = f'collected/mistakes/{sub_dir}/'
    shutil.move(f'collected/normal/{sub_dir}/{base_name}_a.png', destination)
    shutil.move(f'collected/normal/{sub_dir}/{base_name}_b.png', destination)
    push_files_async()


class PetFlaggingCallback(gr.FlaggingCallback):

    def setup(self, components, flagging_dir: str):
        print(components)

    def flag(self, flag_data, flag_option=None, flag_index=None, username=None):
        _, _, label, base_name = flag_data
        move_to_flagged(base_name, label['label'])


demo = gr.Interface(
    title="Dog Recognition",
    description="Model that compares two images and identify if the belong to the same or different dog.",
    fn=predict_diff,
    inputs=[gr.Image(label="Image A", type="pil"), gr.Image(label="Image B", type="pil")],
    outputs=["label", gr.Text(visible=False)],
    flagging_callback=PetFlaggingCallback()
)

demo.queue()

if __name__ == "__main__":
    model_cls = torch.jit.load('model_scripted.pt')
    feat_extractor = VitLaionFeatureExtractor()
    processor = feat_extractor.transforms
    ds_manager = HFPetDatasetManager(dataset_name, hf_token=HF_API_TOKEN, queue=ds_manager_queue)
    ds_manager.daemon = True
    ds_manager.start()
    model_cls.to(device)
    feat_extractor.to(device)
    model_cls.eval()
    feat_extractor.eval()
    demo.launch(share=True)
