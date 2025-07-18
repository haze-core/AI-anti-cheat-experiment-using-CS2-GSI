using CounterStrike2GSI;
using CounterStrike2GSI.EventMessages;
using CounterStrike2GSI.Nodes;
using CounterStrike2GSI.Nodes.Helpers;
using System.Numerics;
using System.Threading.Tasks;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


namespace CounterStrike2GSI_Example_program
{
    public interface IActivation
    {
        double Activate(double x);
        double Derivative(double x);
    }

    public class Sigmoid : IActivation
    {
        public double Activate(double x) => 1.0 / (1.0 + Math.Exp(-x));
        public double Derivative(double x)
        {
            double y = Activate(x);
            return y * (1 - y);
        }
    }

    public class NeuralNetwork
    {
        public double[][][] neuron_weights; // [layer][neuron][input weight (importance)]
        public double[][] neuron_biases; // [layer][neuron]
        public double[][] neuron_outputs; // [layer][neuron]
        public IActivation activation;
        public double learningRate;

        public NeuralNetwork(int[] layerChildCount)
        {
            int amountOfLayersExceptDataInputLayer = layerChildCount.Length - 1;
            int amountOfAllLayers = layerChildCount.Length;
            var rand = new Random();

            // Allocate each layer with weights
            // Allocated: [Amount of layers][Amount of neurons - Not allocated][Amount of input weights - Not allocated]
            neuron_weights = new double[amountOfAllLayers][][];
            // Weights aren't used in the input layer
            neuron_weights[0] = null;
            // Allocate each layer with biases
            // Allocated: [Amount of layers][Amount of neurons - Not allocated]
            neuron_biases = new double[amountOfAllLayers][];
            // Biases aren't used in the input layer
            neuron_biases[0] = null;
            // Allocate each layer with outputs
            // Allocated: [Amount of layers][Amount of neurons - Not allocated]
            neuron_outputs = new double[amountOfAllLayers][];

            for (int currentLayer = 1; currentLayer < amountOfAllLayers; currentLayer++)
            {
                // Get neuron count (layer child count) of the current layer and get input count
                // of the current layer which is the neuron count of the previous layer
                int neuronCount = layerChildCount[currentLayer];
                int inputCount = layerChildCount[currentLayer - 1];

                // Allocate the neurons that have outputs, weights and biases (excluding input data layer)
                // Input data layers outputs are allocated outside the loop
                neuron_outputs[currentLayer] = new double[neuronCount];
                neuron_weights[currentLayer] = new double[neuronCount][];
                neuron_biases[currentLayer] = new double[neuronCount];

                // For every neuron in the current layer...
                for (int currentNeuron = 0; currentNeuron < neuronCount; currentNeuron++)
                {
                    // Allocate input weights: one for each input from the previous layer (the importance of the input)
                    neuron_weights[currentLayer][currentNeuron] = new double[inputCount];

                    // Randomize all input weights between 1 and -1
                    for (int currentInput = 0; currentInput < inputCount; currentInput++)
                    {
                        neuron_weights[currentLayer][currentNeuron][currentInput] = rand.NextDouble() * 2 - 1;
                    }

                    // Randomize the biases of each neuron between 1 and -1
                    neuron_biases[currentLayer][currentNeuron] = rand.NextDouble() * 2 - 1;
                }
            }

            // Finally, allocate space for the outputs of the input layer, which is at index 0
            // Equals to child count of the layer at index 0    
            neuron_outputs[0] = new double[layerChildCount[0]];
        }

        public double[] Predict(double[] input)
        {
            // Copy input to the input layer's outputs
            for (int i = 0; i < input.Length; i++)
                neuron_outputs[0][i] = input[i];

            // Loop through each layer (starting from 1, since layer 0 is input)
            for (int layer = 1; layer < neuron_outputs.Length; layer++)
            {
                for (int neuron = 0; neuron < neuron_outputs[layer].Length; neuron++)
                {
                    double sum = 0.0;

                    // Weighted sum of previous layer outputs
                    for (int inputIndex = 0; inputIndex < neuron_outputs[layer - 1].Length; inputIndex++)
                    {
                        sum += neuron_outputs[layer - 1][inputIndex] * neuron_weights[layer][neuron][inputIndex];
                    }

                    sum += neuron_biases[layer][neuron]; // Add bias

                    neuron_outputs[layer][neuron] = activation.Activate(sum); // Activation function
                }
            }

            // Return final layer's outputs
            return neuron_outputs[neuron_outputs.Length - 1];
        }


        public void Train(double[] input, double[] target, double learningRate)
        {
            // Forward propagation
            double[] prediction = Predict(input);

            // Backward propagation
            double[][] errors = new double[neuron_outputs.Length][];
            double[][] gradients = new double[neuron_outputs.Length][];

            int lastLayer = neuron_outputs.Length - 1;

            // Output layer error = prediction - target
            errors[lastLayer] = new double[prediction.Length];
            gradients[lastLayer] = new double[prediction.Length];
            for (int i = 0; i < prediction.Length; i++)
            {
                double error = target[i] - prediction[i];
                errors[lastLayer][i] = error;

                gradients[lastLayer][i] = error * activation.Derivative(neuron_outputs[lastLayer][i]);
            }

            // Backpropagate error to hidden layers
            for (int layer = lastLayer - 1; layer >= 1; layer--)
            {
                int neuronCount = neuron_outputs[layer].Length;
                int nextNeuronCount = neuron_outputs[layer + 1].Length;

                errors[layer] = new double[neuronCount];
                gradients[layer] = new double[neuronCount];

                for (int i = 0; i < neuronCount; i++)
                {
                    double error = 0.0;

                    // Sum of next layer's weights * gradient
                    for (int j = 0; j < nextNeuronCount; j++)
                    {
                        error += neuron_weights[layer + 1][j][i] * gradients[layer + 1][j];
                    }

                    errors[layer][i] = error;
                    gradients[layer][i] = error * activation.Derivative(neuron_outputs[layer][i]);
                }
            }

            // Update weights and biases
            for (int layer = 1; layer < neuron_outputs.Length; layer++)
            {
                int neuronCount = neuron_outputs[layer].Length;
                int inputCount = neuron_outputs[layer - 1].Length;

                for (int neuron = 0; neuron < neuronCount; neuron++)
                {
                    for (int inputIndex = 0; inputIndex < inputCount; inputIndex++)
                    {
                        double delta = gradients[layer][neuron] * neuron_outputs[layer - 1][inputIndex];
                        neuron_weights[layer][neuron][inputIndex] += learningRate * delta;
                    }

                    neuron_biases[layer][neuron] += learningRate * gradients[layer][neuron];
                }
            }
        }
    }

    class Program
    {
        static GameStateListener? _gsl;
        static string playerName = "";
        static DateTime enemySeenAt, shotAt;
        static string targetName = "";
        static double[] reactionTimes = new double[6];
        static int total = 0;
        static bool dataCollected = false;

        static void Main(string[] args)
        {
            // Create a new Neural Network with 6 inputs, 6 hidden layers and 1 output
            var nn = new NeuralNetwork(new int[] { 6, 6, 1 });
            nn.activation = new Sigmoid();
            nn.learningRate = 0.1;
            int epochs = 0;

            // Dataset
            double[][] inputs = new double[][]
            {
                new double[] { 6f, 1f, 5f, 40f, 220f, 240f },
                new double[] { 180f, 200f, 220f, 270f, 250f, 300f },
                new double[] { 110f, 180f, 250f, 120f, 50f, 100f }
            };

            double[][] targets = new double[][]
            {
                new double[] { 1 },
                new double[] { 0 },
                new double[] { 0.5 }
            };

            // Training
            for (int epoch = 0; epoch < 256000; epoch++)
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    nn.Train(inputs[i], targets[i], nn.learningRate);
                }
                epochs++;
            }

            Console.WriteLine($"Training complete. Total epochs/steps: {epochs}");

            _gsl = new GameStateListener(4000);

            Console.WriteLine("Enter the name of the player you want to examine:");
            playerName = Console.ReadLine() ?? "";

            _gsl.NewGameState += OnNewGameState;
            //_gsl.PlayerTookDamage += OnTookDamage;
            _gsl.KillFeed += OnKillFeed;

            if (!_gsl.Start())
            {
                Console.WriteLine("GameStateListener could not start. Try running this program as Administrator. Exiting.");
                Console.ReadLine();
                Environment.Exit(0);
            }

            Console.WriteLine("Listening for game integration calls...");

            Console.WriteLine("Press ESC to quit");
            do
            {
                while (!Console.KeyAvailable)
                {
                    if (dataCollected)
                    {
                        var output = nn.Predict(reactionTimes);
                        if (output[0] > 0.45 && output[0] < 0.85)
                        {
                            Console.WriteLine("Final verdict: Irregular reaction time.");
                        }
                        else
                        {
                            Console.WriteLine($"Final verdict: Probability of cheating: {Math.Round(output[0] * 100, 2)} %");
                        }
                    }
                    Thread.Sleep(1000);
                }
            } while (Console.ReadKey(true).Key != ConsoleKey.Escape);
        }

        private static float DotProduct(Vector3D a, Vector3D b)
        {
            return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
        }

        private static Vector3D SubtractVectors(Vector3D a, Vector3D b)
        {
            return new Vector3D(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
        }

        private static Vector3D NormalizeVector(Vector3D a)
        {
            float magnitude = (float)Math.Sqrt(a.X * a.X + a.Y * a.Y + a.Z * a.Z);
            if (magnitude == 0) return new Vector3D(0, 0, 0);
            return new Vector3D(a.X / magnitude, a.Y / magnitude, a.Z / magnitude);
        }

        private static bool IsVector3DNull(Vector3D v)
        {
            return float.IsNaN(v.X) || float.IsNaN(v.Y) || float.IsNaN(v.Z) ||
                   (v.X == 0f && v.Y == 0f && v.Z == 0f);
        }

        private static void OnTookDamage(PlayerTookDamage playerTookDamage)
        {
            if (playerTookDamage.Player.Name == targetName)
            {
                shotAt = DateTime.Now;
                double reactionTime = (shotAt - enemySeenAt).TotalMilliseconds;
                if (reactionTime > 300.0f)
                   return;

                if (total < 6)
                {
                    reactionTimes[total] = reactionTime;
                    Console.WriteLine($"{reactionTimes[total]}");
                    total++;
                }
            }
        }

        private static void OnKillFeed(KillFeed killFeed)
        {
            if (killFeed.Killer.Name == playerName)
            {
                shotAt = DateTime.Now;
                double reactionTime = (shotAt - enemySeenAt).TotalMilliseconds;
                if (reactionTime > 300.0f)
                    return;

                if (total < 6)
                {
                    reactionTimes[total] = reactionTime;
                    Console.WriteLine($"{reactionTimes[total]}");
                    total++;
                }

                if (total == 6)
                {
                    dataCollected = true;
                }
            }
        }

        private static void OnNewGameState(CounterStrike2GSI.GameState state)
        {
            Player? examinedPlayer = null;

            foreach (var player in state.AllPlayers.Values)
            {
                if (player.Name == playerName)
                {
                    examinedPlayer = player;
                    break;
                }
            }

            if (examinedPlayer == null || IsVector3DNull(examinedPlayer.Position) || IsVector3DNull(examinedPlayer.ForwardDirection))
                return;

            Vector3D examinedPlayerPos = examinedPlayer.Position;
            Vector3D examinedPlayerForwardDir = NormalizeVector(examinedPlayer.ForwardDirection);

            foreach (var other in state.AllPlayers.Values)
            {
                if (other == null || IsVector3DNull(other.Position))
                    continue;

                if (other.Team == examinedPlayer.Team || other.Name == playerName || other.State.Health == 0)
                    continue;
                
                Vector3D enemyPos = other.Position;
                Vector3D vectorToEnemy = NormalizeVector(SubtractVectors(enemyPos, examinedPlayerPos));

                float dot = DotProduct(examinedPlayerForwardDir, vectorToEnemy);

                if (dot > 0.999)
                {
                    enemySeenAt = DateTime.Now;
                    targetName = other.Name;
                    //Console.WriteLine($"Spotted enemy: {targetName}");
                }
            }
        }
    }
}
