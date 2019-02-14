using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NN_Demo.Activators;
using NN_Demo.Model_File_Save;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_Demo
{
    public class NeuralNet
    {
        public List<Neuron>[] Structure;

        public int Epoch;
        public double LearningRate { get; set; }
        public double Momentum { get; set; }

        public double TotalError { get; set; }

        public List<Sample> TrainingData;

        public double TolerableError { get; set; }

        public NeuralNet()
        {

        }

        public NeuralNet(IActivator[] Activations, int[] Hyperparameters)
        {
            TotalError = 1d;

            if (Activations.Length != Hyperparameters.Length)
                throw new Exception("number of activations isn't equal to layers number");

            //Init array
            Structure = new List<Neuron>[Hyperparameters.Count()];


            //Setup the first layer and its neurons
            Structure[0] = new List<Neuron>();
            for (int i = 0; i < Hyperparameters[0]; i++)
            {
                Structure[0].Add(new Neuron() { LayerIndex = 0, NeuronIndex = i, Activation = Activations[0], NNref = this });
            }

            Structure[0].Add(new Neuron() { LayerIndex = 0, NeuronIndex = Hyperparameters[0], Activation = new BiasActivation(), NNref = this });

            //Wire the rest of neurons with their linkers
            for (int i = 1; i < Structure.Count(); i++)
            {
                Structure[i] = new List<Neuron>();

                for (int j = 0; j < Hyperparameters[i]; j++)
                {
                    var neuron = new Neuron() { LayerIndex = i, NeuronIndex = j, Activation = Activations[i], NNref = this };

                    for (int k = 0; k < Structure[i - 1].Count; k++)
                    {
                        //Create link
                        var Link = new WeightLink() { Left = Structure[i - 1][k], Right = neuron };
                        Link.RandomizeWeight();

                        //Link previous neuron
                        Structure[i - 1][k].NextLinks.Add(Link);
                        neuron.PreviousLinks.Add(Link);
                    }

                    Structure[i].Add(neuron);

                }

                if (i != Structure.Count() - 1)
                {
                    Structure[i].Add(new Neuron() { LayerIndex = i, NeuronIndex = Hyperparameters[i], Activation = new BiasActivation(), NNref = this });
                }
            }

        }


        public void ForwadPropagation()
        {
            foreach (var layer in Structure.Skip(1))
            {
                Parallel.ForEach(layer, (neuron) =>
               {
                   neuron.ComputeInputs();
                   neuron.ComputeActivatedOutput();
               });


            }
        }

        public void SetInputs(IList<double> Inputs)
        {
            if (Structure[0].Count - 1 != Inputs.Count)
                throw new Exception("Inputs size error");

            for (int i = 0; i < Structure[0].Count - 1; i++)
            {
                Structure[0][i].ActivatedOutput = Inputs[i];
            }
        }


        public void AdjustWeights()
        {
            for (int i = Structure.Count() - 1; i >= 0; i--)
            {
                for (int j = 0; j < Structure[i].Count; j++)
                {
                    for (int k = 0; k < Structure[i][j].NextLinks.Count; k++)
                    {
                        Structure[i][j].NextLinks[k].AdjustWeight(LearningRate, Momentum);
                    }
                }
            }
        }

        void ComputeErrorOutput(IList<double> Ideal, ref double GlobalError)
        {
            for (int i = 0; i < Structure[Structure.Count() - 1].Count; i++)
            {
                Structure[Structure.Count() - 1][i].ComputeErrorOutput(Ideal[i], ref GlobalError);
            }
        }

        void CalculateGammaOutput()
        {

            foreach (var layer in Structure.Last())
            {
                for (int j = 0; j < layer.PreviousLinks.Count; j++)
                {
                    layer.ComputeGammaOutput();
                }
            }
        }

        void CalculateDeltaWeightsOutput()
        {

            foreach (var layer in Structure.Last())
            {
                for (int j = 0; j < layer.PreviousLinks.Count; j++)
                {
                    layer.PreviousLinks[j].AdjustDeltaOutput();
                }
            }
        }

        void BackPropOutput(IList<double> Expected, ref double GlobalError)
        {
            ComputeErrorOutput(Expected, ref GlobalError);
            CalculateGammaOutput();
            CalculateDeltaWeightsOutput();
        }

        void BackpropHidden()
        {
            foreach (var layer in Structure.Reverse().Skip(1))
            {
                Parallel.ForEach(layer, (neuron) =>
                {
                    {
                        neuron.Gamma = 0;
                        foreach (var link in neuron.NextLinks)
                        {
                            neuron.Gamma += link.Right.Gamma * link.Weight;
                        }

                        neuron.Gamma *= neuron.ActivatedOutput;
                    }
                });

                Parallel.ForEach(layer, (neuron) =>
                 {
                     {
                         foreach (var link in neuron.PreviousLinks)
                         {
                             link.Delta = link.Right.Gamma * link.Left.ActivatedOutput;
                         }
                     }
                 });


            }
        }

        public void BackPropagation(IList<double> Expected, ref double GlobalError)
        {
            BackPropOutput(Expected, ref GlobalError);
            BackpropHidden();
        }

        public void DescribeError()
        {
            Console.WriteLine("Error MS : " + TotalError);
        }

        public void DescribeNN()
        {

            foreach (var layer in Structure)
            {
                foreach (var neuron in layer)
                {
                    Console.WriteLine(neuron);
                }

                Console.WriteLine("_______________________________________________________");
            }



        }

        public void ExportWeights(string path)
        {

            var settings = new JsonSerializerSettings()
            {
                TypeNameHandling = TypeNameHandling.All
            };

            //settings.Converters.Add(new IActivatorConverter());

            var WeightsData = new List<WeightLink>();

            foreach(var layer in Structure.Skip(1) )
            {
                foreach(var neuron in layer)
                {
                    foreach( var link in neuron.PreviousLinks)
                    {
                        WeightsData.Add(link);
                    }
                }
            }

            var Activators = Structure.Select(x => x.First()).Select(x => x.Activation).ToList();

            var Hyperparameters = Structure.Select((l, index) =>
            {
                if (index != Structure.Count() - 1)
                    return l.Count - 1;
                else
                    return l.Count;
            }).ToList();

            var Data = new NNData { Details = Hyperparameters, Activators = Activators,  LearningRate = LearningRate, Momentum = Momentum, Weights = WeightsData  };

            var jsonString = JsonConvert.SerializeObject(Data , settings);

            if (File.Exists(path))
            {
                File.Delete(path);
            }


            // Create a file to write to.
            using (StreamWriter sw = File.CreateText(path))
            {
                sw.Write(jsonString);
            }
        }

        public static NeuralNet ImportNN(string path)
        {
            var settings = new JsonSerializerSettings()
            {
                TypeNameHandling = TypeNameHandling.All
            };

            var data = File.ReadAllText(path);

            var result = JsonConvert.DeserializeObject<NNData>(data, settings);

            return DataToNN(result);

        }

        private static NeuralNet DataToNN(NNData data)
        {
            var LinksQueue = new Queue<WeightLink>(data.Weights);

            NeuralNet NN = new NeuralNet();
            //Init array


            NN.Structure = new List<Neuron>[data.Details.Count()];


            //Setup the first layer and its neurons
            NN.Structure[0] = new List<Neuron>();
            for (int i = 0; i < data.Details[0]; i++)
            {
                NN.Structure[0].Add(new Neuron() { LayerIndex = 0, NeuronIndex = i, Activation = data.Activators[0], NNref = NN });
            }

            NN.Structure[0].Add(new Neuron() { LayerIndex = 0, NeuronIndex = data.Details[0], Activation = new BiasActivation(), NNref = NN });

            //Wire the rest of neurons with their linkers
            for (int i = 1; i < NN.Structure.Count(); i++)
            {
                NN.Structure[i] = new List<Neuron>();

                for (int j = 0; j < data.Details[i]; j++)
                {
                    var neuron = new Neuron() { LayerIndex = i, NeuronIndex = j, Activation = data.Activators[i], NNref = NN };

                    for (int k = 0; k < NN.Structure[i - 1].Count; k++)
                    {
                        var dequeued = LinksQueue.Dequeue();

                        //Create link
                        var Link = new WeightLink() { Left = NN.Structure[i - 1][k], Right = neuron , Weight = dequeued.Weight  };

                        //Link previous neuron
                        NN.Structure[i - 1][k].NextLinks.Add(Link);
                        neuron.PreviousLinks.Add(Link);
                    }

                    NN.Structure[i].Add(neuron);

                }

                if (i != NN.Structure.Count() - 1)
                {
                    NN.Structure[i].Add(new Neuron() { LayerIndex = i, NeuronIndex = data.Details[i], Activation = new BiasActivation(), NNref = NN });
                }
            }

            Console.WriteLine("Weights left :" + LinksQueue.Count );

            return NN;
        }
    }
}
