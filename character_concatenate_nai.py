class CharacterConcatenateNAI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "character1": ("CHARACTER_NAI", {"forceInput": True}),
            },
            "optional": {
                "character2": ("CHARACTER_NAI", {"forceInput": False}),
                "character3": ("CHARACTER_NAI", {"forceInput": False}),
                "character4": ("CHARACTER_NAI", {"forceInput": False}),
                "character5": ("CHARACTER_NAI", {"forceInput": False}),
                "character6": ("CHARACTER_NAI", {"forceInput": False}),
            }
        }

    RETURN_TYPES = ("CHARACTER_LIST_NAI",)
    FUNCTION = "concat"
    CATEGORY = "NovelAI/v4"

    def concat(self, character1, character2=None, character3=None, character4=None, character5=None, character6=None):
        # Convert inputs to list and exclude "None"
        characters = [character1, character2, character3, character4, character5, character6]
        characters = [c for c in characters if c is not None]
        return (characters,)
