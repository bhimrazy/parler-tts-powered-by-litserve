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
        model_name = "parler-tts/parler-tts-mini-v1"
        # "eager" or "sdpa" or "flash_attention_2"
        attn_implementation = "flash_attention_2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
        ).to(device)

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
