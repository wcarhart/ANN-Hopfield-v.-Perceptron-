/*
 * COMP 380:	Neural Networks
 * Project 1:	Perceptron Neural Network
 * Authors:		Will Carhart, Chaise Brown
 * Date:		February 28th, 2017
 * Summary:		This program implements a perceptron learning algorithm to
 * 				classify specific letters in multiple fonts.
 */

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class fpPerceptron {
	public static void main(String[] args) {
		// file IO variables
		Scanner kb = new Scanner(System.in);
		String trainingfile = null, trainedweights = null;
		String dummy;
		BufferedReader br = null;
		PrintWriter pw = null;

		while (true) {
			System.out
					.println("Welcome to my first neural network - A Perceptron Net!");
			System.out
					.println("Enter 1 to train using a training data file, enter 2 to train using a trained weight settings data file:");
			int trainingMethod = kb.nextInt();
			dummy = kb.nextLine();
			switch (trainingMethod) {
			case 1:
				boolean validFile = false;
				while (!validFile) {
					System.out.println("Enter the training data file name: ");
					trainingfile = kb.nextLine();
					try {
						br = new BufferedReader(new FileReader(trainingfile));
						validFile = true;
						br.close();
					} catch (IOException e) {
						System.err.print(trainingfile + " not found\n");
					}
				}
				train(trainingfile);
				System.out
						.println("Enter the trained weight settings input data file name: ");
				trainedweights = kb.nextLine();
				test(trainedweights);
				break;
			case 2:
				System.out
						.println("Enter the trained weight settings input data file name: ");
				trainedweights = kb.nextLine();
				test(trainedweights);
				break;
			}
		}
	}

	/*
	 * Trains the neural network using a specific input file
	 * 
	 * @param trainingfile the name of the input training file
	 */
	public static void train(String trainingfile) {
		System.out.println("Beginning training...");

		// IO Variables
		BufferedReader br = null;
		String currentLine, trainedweights;
		String[] lineElements = null;
		int input_dim = 0;
		int output_dim = 0;
		Scanner kb = new Scanner(System.in);

		// Training Variables
		boolean converged = false;
		double alpha;
		double theta;
		int MAX_EPOCHS;

		try {
			br = new BufferedReader(new FileReader(trainingfile));
			currentLine = br.readLine();
			lineElements = currentLine.split(",");
			StringBuilder sb = new StringBuilder(lineElements[0]);
			sb.deleteCharAt(0);
			sb.deleteCharAt(0);
			sb.deleteCharAt(0);
			String newStr = sb.toString();
			input_dim = Integer.parseInt(newStr);
			currentLine = br.readLine();
			lineElements = currentLine.split(",");
			output_dim = Integer.parseInt(lineElements[0]);
			currentLine = br.readLine();
			lineElements = currentLine.split(",");
			br.readLine();
		} catch (IOException e) {
			System.err.print("Error in I/O");
			System.exit(1);
		}

		/*
		 * first element of each first dimension array is the bias first
		 * dimension is the row + column data second dimension is the pattern we
		 * are trying to recognize (output_dim)
		 */
		System.out
				.println("Enter 0 to initialize weights to 0, enter 1 to intialize weights to random values between -0.5 and 0.5.");
		int input = kb.nextInt();
		Random rand = new Random();
		double n = 1.5;
		double num;
		double my_num;
		double[][] weights = new double[output_dim][input_dim + 1];
		if (input == 0) {
			for (int a = 0; a < output_dim; a++) {
				for (int b = 0; b < input_dim + 1; b++) {
					weights[a][b] = 0;
				}
			}
		} else {
			for (int a = 0; a < output_dim; a++) {
				for (int b = 0; b < input_dim + 1; b++) {
					num = rand.nextDouble() % n;
					my_num = -0.5 + num;
					weights[a][b] = my_num;
				}
			}
		}

		String dummy = kb.nextLine();
		System.out.println("Enter the maximum number of training epochs: ");
		MAX_EPOCHS = kb.nextInt();
		int epochs = 0; // the number of epochs that have occurred

		dummy = kb.nextLine();
		System.out
				.println("Enter a file name to save the trained weight settings: ");
		trainedweights = kb.nextLine();

		System.out
				.println("Enter the learning rate alpha from 0 to 1 but not including 0: ");
		alpha = kb.nextDouble();
		boolean run = true;
		while (run) {
			if (alpha == 0) {
				System.out
						.println("Alpha should not be zero. Please try again.");
				System.out
						.println("Enter the learning rate alpha from 0 to 1 but not including 0: ");
				alpha = kb.nextDouble();
			} else
				run = false;
		}

		System.out.println("Enter the threshold theta: ");
		theta = kb.nextDouble();

		ArrayList<Integer> trainingData = null; // training data collected from
												// training file
		int target; // target
		int yin; // inputs to the af
		int y; // outputs of the af
		double[][] w_old = new double[output_dim][input_dim + 1];

		boolean changed = false;

		while (!converged && epochs < MAX_EPOCHS) {

			// reading in training set
			try {
				while ((currentLine = br.readLine()) != null) {
					trainingData = new ArrayList<Integer>();
					int[] targets = new int[output_dim];
					lineElements = currentLine.split(",");
					target = Integer.parseInt(lineElements[0]);
					for (int i = 0; i < targets.length; i++) {
						if (i == target) {
							targets[i] = 1;
						} else {
							targets[i] = -1;
						}
					}
					for (int i = 1; i < lineElements.length; i++) {
						if (Integer.parseInt(lineElements[i]) > 0)
							trainingData.add(1);
						else
							trainingData.add(-1);
					}

					// Training
					yin = 0;
					for (int i = 0; i < output_dim; i++) {
						for (int j = 0; j < input_dim; j++) {
							yin += trainingData.get(j) * weights[i][j + 1];
						}
						yin += weights[i][0];
						y = activationFunction(yin, theta);
						if (y != targets[i]) {
							changed = true;
							for (int j = 0; j < (input_dim + 1); j++) {
								w_old[i][j] = weights[i][j];
							}
							for (int j = 1; j < (input_dim + 1); j++) {
								weights[i][j] = w_old[i][j]
										+ (alpha * targets[i] * trainingData
												.get(j - 1));
							}
							weights[i][0] = w_old[i][0] + (alpha * targets[i]);
						}
					}
				}

				// record
				epochs++;
				if (!changed) {
					converged = true;
				}
				changed = false;
				try {
					br = new BufferedReader(new FileReader(trainingfile));
					br.readLine();
					br.readLine();
					br.readLine();
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				} catch (IOException e) {
					e.printStackTrace();
				}
			} catch (IOException e) {
				System.err.print("Error in I/O");
				System.exit(1);
			}
		}

		System.out.println("Training converged after " + epochs + " epochs.");

		System.out.println("Saving weights to file...");

		saveWeights(input_dim, output_dim, weights, trainedweights);

		System.out.println("Weights saved to file!");
	}

	/*
	 * Implements the deployment of the trained neural network
	 * 
	 * @param trainedweights the name of the file containing the trained weights
	 */
	public static void test(String trainedweights) {

		BufferedReader br;
		Scanner kb = new Scanner(System.in);
		String[] lineElements;
		String testingfile = null, testingresults = null, dummy, currentLine;
		int input_dim = 0, output_dim = 0;
		int num = 0;
		double theta;

		System.out
				.println("Enter 1 to test using a testing file, enter 2 to quit.");
		int answer = kb.nextInt();
		switch (answer) {
		case 1:
			dummy = kb.nextLine();
			System.out.println("Enter the testing data file name.");
			testingfile = kb.nextLine();
			System.out.println("Enter a file name to save the testing results");
			testingresults = kb.nextLine();
			break;
		case 2:
			System.exit(0);
			break;
		}

		// read in weights from file
		double weights[][] = null;
		try {
			br = new BufferedReader(new FileReader(trainedweights));
			currentLine = br.readLine();
			input_dim = Integer.parseInt(currentLine);
			currentLine = br.readLine();
			output_dim = Integer.parseInt(currentLine);
			br.readLine();

			weights = new double[output_dim][input_dim + 1];
			for (int i = 0; i < output_dim; i++) {
				for (int j = 0; j < input_dim + 1; j++) {
					weights[i][j] = Double.parseDouble(br.readLine());
				}
			}
		} catch (FileNotFoundException e) {
			System.err.print("File not found");
			System.exit(1);
		} catch (IOException e) {
			System.err.print("Error in I/O");
			System.exit(1);
		}

		System.out.println("Please enter threshold theta: ");
		theta = kb.nextDouble();

		// read in test cases
		int yin, y;
		ArrayList<Integer> testingData = null;
		int target;

		try {
			br = new BufferedReader(new FileReader(testingfile));
			currentLine = br.readLine();
			lineElements = currentLine.split(",");
			StringBuilder sb = new StringBuilder(lineElements[0]);
			sb.deleteCharAt(0);
			sb.deleteCharAt(0);
			sb.deleteCharAt(0);
			String newStr = sb.toString();
			input_dim = Integer.parseInt(newStr);
			currentLine = br.readLine();
			lineElements = currentLine.split(",");
			output_dim = Integer.parseInt(lineElements[0]);
			currentLine = br.readLine();
			lineElements = currentLine.split(",");
			br.readLine();

			while ((currentLine = br.readLine()) != null) {
				testingData = new ArrayList<Integer>();
				int[] targets = new int[output_dim];
				lineElements = currentLine.split(",");
				target = Integer.parseInt(lineElements[0]);
				for (int i = 0; i < targets.length; i++) {
					if (i == target) {
						targets[i] = 1;
					} else {
						targets[i] = -1;
					}
				}
				for (int i = 0; i < lineElements.length; i++) {
					if (Integer.parseInt(lineElements[i]) > 0)
						testingData.add(1);
					else
						testingData.add(-1);
				}

				int[] results = new int[output_dim];
				yin = 0;

				for (int i = 0; i < output_dim; i++) {
					for (int j = 0; j < input_dim; j++) {
						yin += testingData.get(j) * weights[i][j + 1];
					}
					yin += weights[i][0];
					y = activationFunction(yin, theta);
					results[i] = y;
				}
				output(results, targets, output_dim, testingresults, num);
				num++;
			}
		} catch (FileNotFoundException e) {
			System.err.print("File not found");
			System.exit(1);
		} catch (IOException e) {
			System.err.print("Error in I/O");
			System.exit(1);
		}
		
		System.out.println("Computing statistics...");
		statistics(testingresults);
	}

	
	/*
	 * Outputs the classification of the perceptron to a file
	 * 
	 * @param results the actual results of the perceptron
	 * 
	 * @param target the expected results of the perceptron
	 * 
	 * @param output_dim the number of output dimensions
	 */
	public static void output(int[] results, int[] targets, int output_dim,
			String testingresults, int num) {
		if (num == 0) {
			FileWriter fwOb;
			PrintWriter pwOb;
			try {
				fwOb = new FileWriter(testingresults, false);
				pwOb = new PrintWriter(fwOb, false);
				pwOb.flush();
				pwOb.close();
				fwOb.close();
			} catch (FileNotFoundException e) {

			} catch (IOException e) {

			}
		}

		int index = 0;
		int r, t = 0;
		char goal = 0;
		boolean found = false;

		while (!found && index < output_dim) {
			if (results[index] == 1) {
				found = true;
			}
			if (!(index + 1 > output_dim)) {
				if (!found) {
					index++;
				}
			} else {
				break;
			}
		}
		r = index;
		if (index == output_dim) {
			r = -1;
		}

		found = false;
		index = 0;
		while (!found) {
			if (targets[index] == 1) {
				found = true;
			}
			index++;
		}
		t = index - 1;
		
		if(r != -1){
			if(results[r] == targets[t]) {
				r = t;
			}	
		}

		FileWriter pw = null;
		try {
			pw = new FileWriter(testingresults, true);
			pw.write("Actual Output: " + t + "\n");
			for (int i = 0; i < output_dim; i++) {
				pw.write(targets[i] + " ");
			}

			pw.write("\nClassified Output: " + r + "\n");
			for (int i = 0; i < output_dim; i++) {
				pw.write(results[i] + " ");
			}
			pw.write("\n\n");
			pw.close();
			
		} catch (FileNotFoundException e) {
			System.err.print("File not found");
			System.exit(1);
		} catch (IOException e) {
			System.err.print("Error in I/O");
			System.exit(1);
		}
	}
	
	public static void statistics(String testingresults) {
		BufferedReader br;
		String currentLine;
		String[] lineElements;
		double correctClassifications = 0;
		double totalClassifications = 0;

		try {
			br = new BufferedReader(new FileReader(testingresults));
			while ((currentLine = br.readLine()) != null) {
				lineElements = currentLine.split(" ");
				int actualNum = Integer.parseInt(lineElements[2]);
				br.readLine();
				currentLine = br.readLine();
				lineElements = currentLine.split(" ");
				int classifiedNum = Integer.parseInt(lineElements[2]);
				if (actualNum == classifiedNum) {
					correctClassifications++;
				}
				totalClassifications++;
				br.readLine();
				br.readLine();
			}
		} catch (IOException e) {
			System.out.println("Error in I/O");
			System.exit(1);
		}

		double accurateClassifications = correctClassifications
				/ totalClassifications;
		
		System.out.println("Total Correct Classifications: " + correctClassifications);
		System.out.println("Total Classifications: " + totalClassifications);
		System.out.println("The perceptron classified "
				+ (accurateClassifications * 100)
				+ "% of the testing patterns correctly.");

	}

	/*
	 * Uses the input to a neuron and a threshold to determine the output of a
	 * neuron
	 * 
	 * @param yin the input to the neuron
	 * 
	 * @param threshold the threshold value that when achieved will cause the
	 * neuron to fire
	 * 
	 * @return the output of the activation function (bipolar)
	 */
	public static int activationFunction(int yin, double threshold) {
		int toReturn = 0;
		if (yin > threshold) {
			toReturn = 1;
		} else if (yin < threshold) {
			toReturn = -1;
		}
		return toReturn;
	}
	
	/*
	 * Saves the weights to an output file
	 * 
	 * @param input_dim the number of inputs to each output
	 * 
	 * @param output_dim the number of output dimensions
	 * 
	 * @param weights the list of weights to save
	 * 
	 * @param destination the name of the output file
	 */
	public static void saveWeights(int input_dim, int output_dim,
			double[][] weights, String destination) {
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(destination, "UTF-8");
			pw.println(input_dim);
			pw.println(output_dim + "\n");
			for (int i = 0; i < output_dim; i++) {
				for (int j = 0; j < input_dim + 1; j++) {
					pw.println(weights[i][j]);
				}
			}
		} catch (FileNotFoundException e) {
			System.err.print("File not found");
			System.exit(1);
		} catch (IOException e) {
			e.printStackTrace();
		}
		pw.close();
	}
}
