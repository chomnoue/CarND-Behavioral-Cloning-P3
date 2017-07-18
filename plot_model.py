from keras.utils import plot_model
from keras.models import load_model
import sys

if __name__ == "__main__":
	name=sys.argv[1]
	model = load_model(name)
	plot_model(model, to_file=name[:-3]+'.png')