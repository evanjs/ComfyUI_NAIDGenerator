class CharacterNAI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive_prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "x": (["A", "B", "C", "D", "E"], {"default": "C"}),
                "y": (["1", "2", "3", "4", "5"], {"default": "3"}),
            }
        }

    RETURN_TYPES = ("CHARACTER_NAI",)
    FUNCTION = "create"
    CATEGORY = "NovelAI/v4"

    def create(self, positive_prompt, x, y, negative_prompt):
        # Convert x ('A'-'E') and y (1-5) to normalized float (0.0~1.0)
        x_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        x_norm = x_map.get(x, 2) / 4.0
        y_norm = (int(y) - 1) / 4.0
        return ({
                    "char_caption": positive_prompt,
                    "centers": [{"x": x_norm, "y": y_norm}],
                    "negative_caption": negative_prompt,
                },)
