from .utils import *

class GetImageMetadata:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_metadata"
    CATEGORY = "NovelAI/utils"

    def get_metadata(self, image):
        return (get_metadata(image),)
