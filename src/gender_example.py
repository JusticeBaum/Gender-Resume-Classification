import pandas as pd

class GenderExample:
    def __init__(self, gender, resume):
        self.gender = gender
        self.resume = resume

    # Prints all fields of self, for debugging purposes
    def display_data(self):
        print(f"Gender: {self.gender}")
        print(f"Resume: {self.resume}")

    # Reads labels from csv in format row 1 = [0 or 1] row 2 = [rest of data]
    def read_gender_examples(file):
        data = pd.read_csv(file, names = ["Gender", "Resume"])
        ex = []

        for index, row in data.iterrows():
            gender = GenderExample(row['Gender'], row['Resume'])
            ex.append(gender)

        return ex