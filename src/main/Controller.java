package main;

public class Controller {

	private final int layers = 1;
	private final int hiddenNeuronsPerLayer = 3;
	private final int inputNeurons = 4;
	private final int outputNeurons = 4;
	private final double errorCriterion = .0005;
	private final double learningRate = .1;
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
			errorGradient[i] = output[i]*(1-output[i])*error[i];
		}
		for(int i = 0; i < currentLayer.pl.getNeurons().length; i++) {
			currentLayer.pl.getNeurons()[i].adjustNeuronWeight(errorGradient[i], learningRate);
		}
		double[] oldGradient = errorGradient;
		while(currentLayer.pl.pl != null) {
			currentLayer = currentLayer.pl;
			int sum;
			double[] hiddenErrorGradient = new double[currentLayer.getSize()];
			for (int i = 0; i < hiddenErrorGradient.length; i++) {
				sum = 0;
				//Error gradient of next neuron times weight sum all outputs of neuron
				Neuron currentNeuron = currentLayer.getNeurons()[i];
				for(int j = 0; j < currentNeuron.weights.length; j++) {
					sum += oldGradient[j] * currentNeuron.weights[j];
				}
				hiddenErrorGradient[i] = sum;
			}
			oldGradient = hiddenErrorGradient;
			for(int i = 0; i < currentLayer.pl.getNeurons().length; i++) {
				currentLayer.pl.getNeurons()[i].adjustNeuronWeight(errorGradient[i], learningRate);
			}
		}
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
			 * Calculate error ^2 of network
			 */
			double sum = 0;
			for(int i = 0; i < error.length; i++) {
				error[i] = desiredOutput[i] - output[i];
				sum += .5 * (Math.pow(error[i], 2));
			}
			trainingSessions++;
			if(trainingSessions % 100000 == 0) {
				System.out.println(trainingSessions);
			}
			/*
			 * Check if the network is smart enough
			 */
			if(sum < errorCriterion) {
				return output;
			} else {
				try {
					Thread.sleep(10);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				runBackPropogation(error, output, outputLayer);
			}
		}


	}
}
