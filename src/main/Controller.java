package main;

public class Controller {

	private static final boolean winnerTakeAll = false;
	private static final int layers = 1;
	private static final int hiddenNeuronsPerLayer = 3;
	private static final int inputNeurons = 4;
	private static final int outputNeurons = 4;
	private static final double errorCriterion = .001;
	private static final double learningRate = .1;
	private double trainingSessions = 0;
	private NeuronLayer inputLayer;
	private NeuronLayer[] hiddenLayers;
	private NeuronLayer outputLayer;

	public Controller() {
		/*
		 * Build the network given the above criteria
		 */
//		inputLayer = new NeuronLayer(inputNeurons, hiddenNeuronsPerLayer);
//		hiddenLayers = new NeuronLayer[layers];
//		if(hiddenLayers.length > 1) {
//			hiddenLayers[0] = new NeuronLayer(hiddenNeuronsPerLayer, inputLayer, hiddenNeuronsPerLayer);
//			for(int i = 1; i < layers - 1; i++) {
//				hiddenLayers[i] = new NeuronLayer(hiddenNeuronsPerLayer, hiddenLayers[i - 1], hiddenNeuronsPerLayer);
//			}
//			hiddenLayers[layers - 1] = new NeuronLayer(hiddenNeuronsPerLayer, hiddenLayers[layers - 2], outputNeurons);
//		} else {
//			hiddenLayers[0] = new NeuronLayer(hiddenNeuronsPerLayer, inputLayer, outputNeurons);
//		}
//		outputLayer = new NeuronLayer(hiddenLayers[hiddenLayers.length - 1], outputNeurons);
		inputLayer = new NeuronLayer(new Neuron[] {new Neuron( 2, 2, 0, .5, .9), new Neuron( 2, 2, 1, .4, 1.0)});
		hiddenLayers = new NeuronLayer[]{ new NeuronLayer(new Neuron[] {new Neuron(inputLayer.getNeurons(), 1, 0, .8, -1.2), new Neuron(inputLayer.getNeurons(), 1, 1, -.1, 1.1)}, inputLayer)};
		outputLayer = new NeuronLayer(new Neuron[] {new Neuron(hiddenLayers[0].getNeurons(), 0.0, 0.3)}, hiddenLayers[0]);
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
			for (int i = 0; i < currentLayer.pl.getNeurons().length; i++) {
				sum = 0;
				for(int j = 0; j < oldGradient.length; j++) {
					sum += oldGradient[j] * currentLayer.pl.getNeurons()[i].oldWeights[j];
				}
				System.out.println(sum);
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

	public static double activationFunctionDerivative(double d) {
		System.out.println(d + " * " + " ( 1 - " + d + ") = " + (d * (1 -d)));
		return d * (1 - d);
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
		while(true) {
			double sum = 0;
			for(int p = 0; p < dataSet.length; p++) {
				/*
				 * Process data through network
				 */
				double[] output = new double[dataSet[p].length];
				double[] error = new double[desiredOutputSet[p].length];
				output = inputLayer.processData(dataSet[p]);
				output = hiddenLayers[0].processData(output);
				output = outputLayer.processData(output);
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
				for(int i = 0; i < error.length; i++) {
					error[i] = desiredOutputSet[p][i] - output[i];
					sum += .5 * (Math.pow(error[i], 2));
				}
				runBackPropogation(error, output, outputLayer);
				try {
					Thread.sleep(10000);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}

			trainingSessions++;
			if(trainingSessions % 10000 == 0) {
				System.out.println(trainingSessions);
				System.out.println(sum / dataSet[0].length);
			}
			if(sum < errorCriterion) {
				return true;
			}


		}


	}
}
