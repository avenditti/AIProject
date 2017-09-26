package main;

public class Controller {

	private static final int layers = 1;
	private static final int hiddenNeuronsPerLayer = 6;
	private static final int inputNeurons = 4;
	private static final int outputNeurons = 4;
	private static final double errorCriterion = .001;
	private static final double learningRate = .2;
	public static final int activationFunction = 1;
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
		/*
		 * Process output neurons
		 */
		double[] errorGradient = new double[output.length];
		for (int i = 0; i < errorGradient.length; i++) {
			errorGradient[i] = activationFunctionDerivative(output[i])*error[i];
			currentLayer.getNeurons()[i].setThreshhold(currentLayer.getNeurons()[i].getThreshhold() + (learningRate * -1 * errorGradient[i]));
		}
		for(int i = 0; i < currentLayer.pl.getNeurons().length; i++) {
			currentLayer.pl.getNeurons()[i].adjustNeuronWeight(errorGradient, learningRate, currentLayer.pl.getNeurons()[i].getPreviousValue());
		}
		double[] oldGradient = errorGradient;
		currentLayer = currentLayer.pl;
		/*
		 * Process hidden layer neurons
		 */
		while(currentLayer.pl != null) {
			double sum;
			double[] hiddenErrorGradient = new double[currentLayer.getSize()];
			for (int i = 0; i < currentLayer.getNeurons().length; i++) {
				sum = 0;
				for(int j = 0; j < currentLayer.getNeurons()[i].oldWeights.length; j++) {
					sum += oldGradient[j] * currentLayer.getNeurons()[i].oldWeights[j];
				}
				hiddenErrorGradient[i] =  activationFunctionDerivative(currentLayer.getNeurons()[i].getPreviousValue()) * sum;
				currentLayer.getNeurons()[i].setThreshhold(currentLayer.getNeurons()[i].getThreshhold() + (learningRate * -1 * hiddenErrorGradient[i]));
			}
			oldGradient = hiddenErrorGradient;
			for(int i = 0; i < currentLayer.pl.getNeurons().length; i++) {
				currentLayer.pl.getNeurons()[i].adjustNeuronWeight(hiddenErrorGradient, learningRate, currentLayer.pl.getNeurons()[i].getPreviousValue());
			}
			currentLayer = currentLayer.pl;
		}
	}

	public static double activationFunctionDerivative(double x) {
		switch(Controller.activationFunction) {
		case 0:
			return x * (1 - x);
		case 1:
			return (2*Neuron.a*Neuron.b*Math.pow(Math.E,(-Neuron.b*x)))/Math.pow((Math.pow(Math.E,(-Neuron.b*x)+1)),2);
		case 2:
			return -2*x*Math.pow(Math.E, -Math.pow(x, 2));
		default:
			return x * (1 - x);
		}

	}

	public boolean[] testNetwork(double[][] dataSet, double[][] desiredOutputSet) {
		boolean[] outputBool = new boolean[dataSet.length];
		double[] data;
		double[] desiredOutput;
		double[] output;
		double[] error;
		for (int k = 0; k < dataSet.length; k++) {
			data = dataSet[k];
			desiredOutput = desiredOutputSet[k];
			output = new double[data.length];
			error = new double[desiredOutput.length];
			output = inputLayer.processData(data);
			output = hiddenLayers[0].processData(output);
			output = outputLayer.processData(output);
			double sum = 0;
			for(int i = 0; i < error.length; i++) {
				error[i] = desiredOutput[i] - output[i];
				sum += .5 * (Math.pow(error[i], 2));
			}
			outputBool[k] = sum < errorCriterion;
		}
		return outputBool;
	}

	public boolean trainNetwork(double[][] dataSet, double[][] desiredOutputSet) {
		double sum;
		double[] output;
		double[] error;
		boolean l = true;
		int h;
		while(true) {
			sum = 0;
			h = (int) (Math.random() * dataSet.length);
			boolean[] tested = new boolean[dataSet.length];
			while(true) {
				/*
				 * Process data through network
				 */
				output = new double[dataSet[h].length];
				error = new double[desiredOutputSet[h].length];
				output = inputLayer.processData(dataSet[h]);
				output = hiddenLayers[0].processData(output);
				output = outputLayer.processData(output);
				/*
				 * Calculate error ^2 of network
				 */
				for(int i = 0; i < error.length; i++) {
					error[i] = desiredOutputSet[h][i] - output[i];
					sum += (Math.pow(error[i], 2));
				}
				runBackPropogation(error, output, outputLayer);
				tested[h] = true;
				l = true;
				for(boolean b : tested) {
					l = l && b;
				}
				if(l) {
					break;
				}
				while(tested[h = (int) (Math.random() * dataSet.length)]);
			}
			sum = sum / dataSet[0].length;
			trainingSessions++;
			if(trainingSessions % 100 == 0) {
				System.out.println(trainingSessions);
				System.out.println(sum);
			}
			if(sum  < errorCriterion) {
				return true;
			}


		}


	}
}
