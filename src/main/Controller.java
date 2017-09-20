package main;

public class Controller {

	private static final boolean winnerTakeAll = false;
	private static final int layers = 1;
	private static final int hiddenNeuronsPerLayer = 3;
	private static final int inputNeurons = 4;
	private static final int outputNeurons = 4;
	private static final double errorCriterion = .00005;
	private static final double learningRate = .2;
	private double trainingSessions = 0;
	private NeuronLayer inputLayer;
	private NeuronLayer[] hiddenLayers;
	private NeuronLayer outputLayer;

	public Controller() {
		/*
		 * Build the network given the above criteria
		 */
		inputLayer = new NeuronLayer(inputNeurons, hiddenNeuronsPerLayer);
		hiddenLayers = new NeuronLayer[layers];
		if(hiddenLayers.length > 1) {
			hiddenLayers[0] = new NeuronLayer(hiddenNeuronsPerLayer, inputLayer, hiddenNeuronsPerLayer);
			for(int i = 1; i < layers - 1; i++) {
				hiddenLayers[i] = new NeuronLayer(hiddenNeuronsPerLayer, hiddenLayers[i - 1], hiddenNeuronsPerLayer);
			}
			hiddenLayers[layers - 1] = new NeuronLayer(hiddenNeuronsPerLayer, hiddenLayers[layers - 2], outputNeurons);
		} else {
			hiddenLayers[0] = new NeuronLayer(hiddenNeuronsPerLayer, inputLayer, outputNeurons);
		}
		outputLayer = new NeuronLayer(hiddenLayers[hiddenLayers.length - 1], outputNeurons);
		/*
		 * Print off network information
		 */
		System.out.println("Input Layer\n" + inputLayer.toString());
		for(int i = 0; i < layers; i++) {
			System.out.println("Hidden Layer " + i + "\n" + hiddenLayers[i].toString());
		}
		System.out.println("Output Layer\n" + outputLayer.toString());
	}

	private void runBackPropogation(double[] error, double[] output, NeuronLayer currentLayer) {
		double[] errorGradient = new double[output.length];
		for (int i = 0; i < errorGradient.length; i++) {
			errorGradient[i] = activationFunction(output[i])*error[i];
		}
		for(int i = 0; i < currentLayer.pl.getNeurons().length; i++) {
			currentLayer.pl.getNeurons()[i].adjustNeuronWeight(errorGradient, learningRate, currentLayer.pl.getNeurons()[i].getPreviousValue());
		}
		double[] oldGradient = errorGradient;
		currentLayer = currentLayer.pl;
		while(currentLayer.pl != null) {
			int sum;
			double[] hiddenErrorGradient = new double[currentLayer.getSize()];
			for (int i = 0; i < hiddenErrorGradient.length; i++) {
				sum = 0;
				//Error gradient of next neuron times weight sum all outputs of neuron
				Neuron currentNeuron = currentLayer.getNeurons()[i];
				for(int j = 0; j < currentNeuron.weights.length; j++) {
					sum += oldGradient[j] * currentNeuron.oldWeights[j];
				}
				hiddenErrorGradient[i] = currentLayer.getNeurons()[i].getPreviousValue() *(1 - currentLayer.getNeurons()[i].getPreviousValue()) * sum;
			}
			oldGradient = hiddenErrorGradient;
			for(int i = 0; i < currentLayer.getNeurons().length; i++) {
				currentLayer.pl.getNeurons()[i].adjustNeuronWeight(hiddenErrorGradient, learningRate, currentLayer.pl.getNeurons()[i].getPreviousValue());
			}
			currentLayer = currentLayer.pl;
		}
	}

	public static double activationFunction(double d) {
		return d * (1 - d);
	}

	public double[] trainNetwork(double[] data, double[] desiredOutput) {
		while(true) {
			/*
			 * Process data through network
			 */
			double[] output = new double[data.length];
			double[] error = new double[desiredOutput.length];
			output = inputLayer.processData(data);
			output = hiddenLayers[0].processData(data);
			output = outputLayer.processData(data);
			/*
			 * If using winner take all then adjust the output
			 */
			if(winnerTakeAll) {
				double max = output[0];
				int j = 0;
				for(int i = 1; i < output.length; i++) {
					if(max < output[i]) {
						max = output[i];
						output[j] = 0;
						j = i;
					} else {
						output[i] = 0;
					}
				}
				output[j] = 1;
			}
			/*
			 * Calculate error ^2 of network
			 */
			double sum = 0;
			for(int i = 0; i < error.length; i++) {
				error[i] = desiredOutput[i] - output[i];
				sum += .5 * (Math.pow(error[i], 2));
			}
			trainingSessions++;
			if(trainingSessions % 5000 == 0) {
				System.out.println(trainingSessions);
				for(double d : output) {
					System.out.println(d);
				}
			}
			/*
			 * Check if the network is smart enough
			 */
			if(sum < errorCriterion) {
				return output;
			} else {
				runBackPropogation(error, output, outputLayer);
			}
		}


	}
}
