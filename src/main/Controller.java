package main;

public class Controller {
	/*
	 * iL -> inputLayer
	 * hl -> hiddenLayers
	 * oL -> oututLayers
	 * iN -> inputNeurons
	 * hnpl -> hiddenNeuronsPerLayer
	 * oN -> outputNeurons
	 */
	private static final int layers = 1;
	private static final int hnpl = 3;
	private static final int iN = 4;
	private static final int oN = 4;
	private static final double errorCriterion = .01;
	private static final double learningRate = .2;
	public static double beta = .95;
	public static int activationFunction = 1;
	private NeuronLayer iL;
	private NeuronLayer[] hL;
	private NeuronLayer oL;

	public Controller(double beta, int function) {
		Controller.beta = beta;
		Controller.activationFunction = function;
		/*
		 * Build the network given the above criteria
		 */
		iL = new NeuronLayer(iN, hnpl);
		hL = new NeuronLayer[layers];
		if(hL.length > 1) {
			hL[0] = new NeuronLayer(hnpl, iL, hnpl);
			for(int i = 1; i < layers - 1; i++) {
				hL[i] = new NeuronLayer(hnpl, hL[i - 1], hnpl);
			}
			hL[layers - 1] = new NeuronLayer(hnpl, hL[layers - 2], oN);
		} else {
			hL[0] = new NeuronLayer(hnpl, iL, oN);
		}
		oL = new NeuronLayer(hL[hL.length - 1], oN);
//		/*
//		 * Print off network information
//		 */
//		System.out.println("Input Layer\n" + iL.toString());
//		for(int i = 0; i < layers; i++) {
//			System.out.println("Hidden Layer " + i + "\n" + hL[i].toString());
//		}
//		System.out.println("Output Layer\n" + oL.toString());
	}

	private void runBackPropogation(double[] error, double[] output, NeuronLayer cL) {
		/*
		 * Process output neurons
		 */
		double[] eg = new double[output.length];
		for (int i = 0; i < eg.length; i++) {
			eg[i] = afd(output[i])*error[i];
			cL.getNeurons()[i].setThreshhold(cL.getNeurons()[i].getThreshhold() + (learningRate * -1 * eg[i]));
		}
		for(int i = 0; i < cL.pl.getNeurons().length; i++) {
			cL.pl.getNeurons()[i].adjustNeuronWeight(eg, learningRate, cL.pl.getNeurons()[i].getPreviousValue());
		}
		cL = cL.pl;
		/*
		 * Process hidden layer neurons
		 */
		while(cL.pl != null) {
			double sum;
			double[] hneg = new double[cL.getSize()];
			for (int i = 0; i < cL.getNeurons().length; i++) {
				sum = 0;
				for(int j = 0; j < cL.getNeurons()[i].oldWeights.length; j++) {
					sum += eg[j] * cL.getNeurons()[i].oldWeights[j];
				}
				hneg[i] =  afd(cL.getNeurons()[i].getPreviousValue()) * sum;
				cL.getNeurons()[i].setThreshhold(cL.getNeurons()[i].getThreshhold() + (learningRate * -1 * hneg[i]));
			}
			eg = hneg;
			for(int i = 0; i < cL.pl.getNeurons().length; i++) {
				cL.pl.getNeurons()[i].adjustNeuronWeight(hneg, learningRate, cL.pl.getNeurons()[i].getPreviousValue());
			}
			cL = cL.pl;
		}
	}

	/*
	 * Activation Function Derivative
	 */
	public static double afd(double x) {
		switch(Controller.activationFunction) {
		case 0:
			return x * (1 - x);
		case 1:
			return 1 - Math.pow(x, 2);
		case 2:
			return -2 * -Math.sqrt(-Math.log(x)) * x;
		default:
			return x * (1 - x);
		}

	}

	public boolean[] testNetwork(double[][] dataSet, double[][] desiredOutputSet) {
		boolean[] outputBool = new boolean[dataSet.length];
		double[] data;
		double[] output;
		for (int k = 0; k < dataSet.length; k++) {
			data = dataSet[k];
			output = new double[data.length];
			output = iL.processData(data);
			output = hL[0].processData(output);
			output = oL.processData(output);
			int max = 0;
			for (int i = 1; i < output.length; i++) {
				if(output[i] > output[max]) {
					output[max] = 0;
					max = i;
				} else {
					output[i] = 0;
				}
			};
			output[max] = 1;
			boolean l = true;
			for (int i = 0; i < output.length; i++) {
				l = l && output[i] == desiredOutputSet[k][i];
			}
			outputBool[k] = l;
		}
		return outputBool;
	}

	public boolean trainNetwork(double[][] dataSet, double[][] desiredOutputSet) {
		double sum;
		double[] output;
		double[] error;
		boolean l = true;
		int h;
		double sum2;
		int trainingSessions = 0;
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
				output = iL.processData(dataSet[h]);
				output = hL[0].processData(output);
				output = oL.processData(output);
				/*
				 * Calculate error ^2 of network
				 */
				sum2 = 0;
				for(int i = 0; i < error.length; i++) {
					error[i] = desiredOutputSet[h][i] - output[i];
					sum2 += (Math.pow(error[i], 2));
				}
				sum += sum2 / error.length;
				runBackPropogation(error, output, oL);
				tested[h] = true;
				l = true;
				for(boolean b : tested)
					l = l && b;
				if(l)
					break;
				while(tested[h = (int) (Math.random() * dataSet.length)]);
			}
			sum = sum / dataSet[0].length;
			trainingSessions++;
			if(trainingSessions % 100 == 0) {
				System.out.printf("%-8d: %.6f \n",trainingSessions, sum);
			}
			if(trainingSessions > 10000) {
				System.out.println("Network will not converge please try training again");
				return false;
			}
			if(sum  < errorCriterion) {
				System.out.printf("%-8d: %.6f \n",trainingSessions, sum);
				return true;
			}


		}


	}
}
