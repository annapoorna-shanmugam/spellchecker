from neuspell import NestedlstmChecker

class LSTMSpellChecker:
    def __init__(self, model_path: str = "neuspell/lstm_spell_checker"):
        self.checker = NestedlstmChecker(model_path=model_path)
        self.checker.from_pretrained()

    def check(self, text: str) -> str:
        errors = []
        refined_text = self.checker.correct_strings(text)
        print("LSTM Spell Checker Results: ", refined_text)
        return errors