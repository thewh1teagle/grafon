from grafon_urdu import Phonemizer

# After the compact model has been trained and approved for publication:
phonemizer = Phonemizer.from_pretrained("grafon-g2p/ur", device="cpu")
print(phonemizer("مجھے اردو زبان سے محبت ہے۔"))
