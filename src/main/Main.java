package main;

public class Main {
	public static void main(String args[]) {
		Controller c = new Controller(Double.parseDouble(args[0]), args[1].equals("ht2") ? 1 : 0);
		if(c.trainNetwork(TestInformation.inputs,TestInformation.desiredOutputs)) {
			int i = 0;
			for(boolean d : c.testNetwork(TestInformation.inputs2, TestInformation.desiredOutputs2)) {
				i++;
				System.out.println("System Test " + i + " " + (d ? "Pass" : "Fail"));
			}
		}
	}
}