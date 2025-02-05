https://drive.google.com/drive/folders/16uAnu8KCX0zipxXOs2mbQ6qYrEMymdtw?usp=sharing
Basic instructions:
-> Download demo video and and put into worker-model folder.
-> Run docker container, and start the instance. well_fitting.mov should be in there
-> run "conda run -n base python /worker-model/detection.py"
	-> If that doesn't work, there is a conda instance called base. Activate it and run detection.py however you're able to
-> Should run the model and provide an output file. You can then export the result into your local machine.