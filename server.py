import io

import litserve as ls
import soundfile as sf
import torch
from fastapi.responses import Response
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

DEFAULT_VOICE_DESCRIPTION = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."


class ParlerTTSAPI(ls.LitAPI):
    def setup(self, device):
        """Initialize the tokenizer, model, and attention mechanism."""
        self.device = device
        model_name = "parler-tts/parler-tts-mini-v1"
        attn_implementation = "flash_attention_2"  # Alternatives: "eager", "sdpa"

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
        ).to(device)

    def decode_request(self, request):
        """Parse input request to extract prompt and optional voice description."""
        prompt = request["prompt"]
        description = request.get("description", DEFAULT_VOICE_DESCRIPTION)

        # Tokenize description and prompt
        description_tokens = self.tokenizer(
            description, return_tensors="pt"
        ).input_ids.to(self.device)
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        return {"input_ids": description_tokens, "prompt_input_ids": prompt_tokens}

    def predict(self, inputs):
        """Generate speech audio using the model."""
        generation = self.model.generate(**inputs)
        audio_arr = generation.cpu().float().numpy().squeeze()

        # Save audio to a buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio_arr, self.model.config.sampling_rate, format="wav")
        return buffer.getvalue()

    def encode_response(self, output):
        """Package the generated audio data into a response."""
        return Response(
            content=output,
            headers={"Content-Type": "audio/wav"},
        )


if __name__ == "__main__":
    # Set up API service and server
    api = ParlerTTSAPI()
    server = ls.LitServer(api, accelerator="auto", api_path="/speech", timeout=100)
    server.run(port=8000)
