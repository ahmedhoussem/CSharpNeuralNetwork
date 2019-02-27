using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NN.Activators;
using NN.ModelFileSave;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN
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

    public NeuralNet(IActivator[] Activations, int[] Hyperparameters, double WeightAmplitude)
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
                    Link.RandomizeWeight(WeightAmplitude);

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
        for (int i = 1; i < Structure.Count(); i++)
        {
            Parallel.For(0, Structure[i].Count, (index) =>
          {
              Structure[i][index].ComputeInputs();
              Structure[i][index].ComputeActivatedOutput();
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

        for (int i = 0; i < Structure.Last().Count; i++)
        {
            Structure.Last()[i].ComputeGammaOutput();
        }
    }

    void CalculateDeltaWeightsOutput()
    {

        for (int i = 0; i < Structure.Last().Count; i++)
        {
            for (int j = 0; j < Structure.Last()[i].PreviousLinks.Count; j++)
            {
                Structure.Last()[i].PreviousLinks[j].AdjustDeltaOutput();
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
        for (int i = Structure.Count() - 2; i >= 0; i--)
        {
            Parallel.For(0, Structure[i].Count, (index) =>
          {
              {
                  Structure[i][index].Gamma = 0;
                  for (int j = 0; j < Structure[i][index].NextLinks.Count; j++)
                  {
                      Structure[i][index].Gamma += Structure[i][index].NextLinks[j].Right.Gamma * Structure[i][index].NextLinks[j].Weight;
                  }

                  Structure[i][index].Gamma *= Structure[i][index].ActivatedOutput;
              }
          });

            Parallel.For(0, Structure[i].Count, (index) =>
            {
                {
                    for (int j = 0; j < Structure[i][index].PreviousLinks.Count; j++)
                    {
                        Structure[i][index].PreviousLinks[j].Delta = Structure[i][index].PreviousLinks[j].Right.Gamma * Structure[i][index].PreviousLinks[j].Left.ActivatedOutput;
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
            TypeNameHandling = TypeNameHandling.Objects
        };

        //settings.Converters.Add(new IActivatorConverter());

        var WeightsData = new List<WeightLink>();

        foreach (var layer in Structure.Skip(1))
        {
            foreach (var neuron in layer)
            {
                foreach (var link in neuron.PreviousLinks)
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

        var Data = new NNData { Details = Hyperparameters, Activators = Activators, LearningRate = LearningRate, Momentum = Momentum, Weights = WeightsData };

        var jsonString = JsonConvert.SerializeObject(Data, settings);

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
            TypeNameHandling = TypeNameHandling.Objects
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
                    var Link = new WeightLink() { Left = NN.Structure[i - 1][k], Right = neuron, Weight = dequeued.Weight };

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

        Console.WriteLine("Weights left :" + LinksQueue.Count);

        return NN;
    }
}
}