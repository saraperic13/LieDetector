import LieDetector

if __name__ == "__main__":
    lieDetector = LieDetector.LieDetector(algorithm='ffnn')
    lieDetector.process()
    lieDetector.destroy()
