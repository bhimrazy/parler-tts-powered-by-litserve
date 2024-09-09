import io

import litserve as ls
import soundfile as sf
import torch
from fastapi.responses import Response
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

DEFAULT_DESCRIPTION = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."


class ParlerTTSAPI(ls.LitAPI):
    def setup(self, device):
        self.device = device
        # need to set padding max length
        self.max_length = 50

        # load model and tokenizer
        model_name = "parler-tts/parler-tts-mini-v1"
        # "eager" or "sdpa" or "flash_attention_2"
        attn_implementation = "flash_attention_2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
        )

        # compile the forward pass
        # self.model.generation_config.cache_implementation = "static"
        # self.model.forward = torch.compile(
        #     self.model.forward, mode="default", fullgraph=True
        # )

        # # warmup
        # self._warmup()

    def _warmup(self):
        # warmup
        print("Warming up...")
        inputs = self.tokenizer(
            "This is for compilation",
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
        ).to(self.device)

        model_kwargs = {
            **inputs,
            "prompt_input_ids": inputs.input_ids,
            "prompt_attention_mask": inputs.attention_mask,
        }

        n_steps = 2
        for _ in range(n_steps):
            _ = self.model.generate(**model_kwargs)
        print("Warmed up!")

    def decode_request(self, request):
        prompt = request["prompt"]
        description = request.get("description", DEFAULT_DESCRIPTION)

        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(
            self.device
        )
        prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        inputs = {"input_ids": input_ids, "prompt_input_ids": prompt_input_ids}

        return inputs

    def predict(self, inputs):
        generation = self.model.generate(**inputs)
        audio_arr = generation.cpu().float().numpy().squeeze()
        buffer = io.BytesIO()
        sf.write(buffer, audio_arr, self.model.config.sampling_rate, format="wav")
        return buffer.getvalue()

    def encode_response(self, output):
        return Response(
            content=output,
            headers={"Content-Type": "audio/wav"},
        )


if __name__ == "__main__":
    api = ParlerTTSAPI()
    server = ls.LitServer(api, accelerator="auto", api_path="/speech", timeout=100)
    server.run(port=8000)
