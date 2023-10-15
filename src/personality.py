class Personality:
    def __init__(self,name: str):
        self.name=name
        self.description = ""
    
    def get_description(self) -> str:
        return self.description

class StandardPersonality(Personality):
    def __init__(self, name: str,description: str):
        super().__init__(name)
        self.name=name
        if not self.name in description:
            self.description = f"You are {self.name} and here is your description : "+ description
        else:
            self.description=description

class AssistedPersonality(Personality):
    def __init__(self,name: str, qualities: list, defaults: list, examples: list):
        super().__init__(name)
        self.name=name
        self.qualities = qualities
        self.defaults = defaults
        self.examples = examples
        self._construct_description()

    def _construct_description(self):
        # Building the description based on the parameters
        qualities_str = ", ".join(self.qualities)
        defaults_str = ", ".join(self.defaults)
        examples_str = "\n - ".join(self.examples)
        
        self.description = f"You are {self.name} You have qualities such as {qualities_str}. Your defaults include {defaults_str}. Here are some examples:\n - {examples_str}"
