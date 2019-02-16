using Newtonsoft.Json;
using NN_Demo.Activators;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NN_Demo
{
    public class Program
    {
        static void Main(string[] args)
        {

            string exportPath = @"PATH_TO_SAVE_NN_FILE\NeuralNetwork.json";

            double GlobalError = 0d;

            var TrainingData = new List<Sample>()
            {
                new Sample {Inputs = new List<double>() { 0 , 0  } , Results = new List<double>() { 1  } },
                new Sample {Inputs = new List<double>() { 1 , 1  } , Results = new List<double>() { 1  } },

                new Sample {Inputs = new List<double>() { 0 , 1  } , Results = new List<double>() { 0  } },
                new Sample {Inputs = new List<double>() { 1 , 0  } , Results = new List<double>() { 0  } },

                new Sample {Inputs = new List<double>() { 0 , 1  } , Results = new List<double>() { 0  } },
                new Sample {Inputs = new List<double>() { 0 , 0  } , Results = new List<double>() { 1  } },

            };

            var TestData = new List<Sample>()
            {
                new Sample {Inputs = new List<double>() { 0 , 0  } , Results = new List<double>() { 1  } },
                new Sample {Inputs = new List<double>() { 1 , 1  } , Results = new List<double>() { 1  } },

                new Sample {Inputs = new List<double>() { 1 , 0  } , Results = new List<double>() { 0  } },
                new Sample {Inputs = new List<double>() { 0 , 1  } , Results = new List<double>() { 0  } },

            };



            NeuralNet NN = new NeuralNet(new IActivator[] { new Sigmoid(), new Sigmoid(), new ReLU() }, new int[] { TrainingData[0].Inputs.Count, 2, TrainingData[0].Results.Count } , 5);

            NN.LearningRate = 0.7d;
            NN.Momentum = 0.4d;

            #region train
            while (NN.TotalError > 0.01d)
            {
                GlobalError = 0;

                foreach (var sample in TrainingData)
                {
                    NN.SetInputs(sample.Inputs);

                    NN.ForwadPropagation();

                    NN.BackPropagation(sample.Results, ref GlobalError);


                    NN.AdjustWeights();



                    NN.DescribeError();

                    // NN.DescribeNN();

                    Console.Clear();
                }

                GlobalError = GlobalError / (NN.Structure.Last().Count * TrainingData.Count);
                NN.TotalError = GlobalError;

                NN.Epoch++;

            }
            #endregion


            #region test
            Console.WriteLine("TEST : ");

            Console.Clear();
            Console.WriteLine("Epochs : " + NN.Epoch);
            foreach (var sample in TestData)
            {
                NN.SetInputs(sample.Inputs);
                NN.ForwadPropagation();
                Console.Write(" Input : (");
                foreach (var input in sample.Inputs)
                {
                    Console.Write($"{ input } , ");
                }
                Console.Write(") , Outputs : (");

                foreach (var outpur in NN.Structure.Last())
                {
                    Console.Write($" { outpur.ActivatedOutput } , ");
                }

                Console.WriteLine(")");
            }

            #endregion
            /*
            #region File import/export
            Console.WriteLine("IMPORTED : ");


            NN.ExportWeights(exportPath);

            var NewNN = NeuralNet.ImportNN(exportPath);

            for (int i = 0; i < TestData.Count; i++)
            {
                NewNN.SetInputs(TestData[i].Inputs);
                NewNN.ForwadPropagation();
                Console.Write(" Input : (");
                foreach (var input in TestData[i].Inputs)
                {
                    Console.Write($"{ input } , ");
                }
                Console.Write(") , Outputs : (");

                foreach (var output in NewNN.Structure.Last())
                {
                    Console.Write($" { output.ActivatedOutput } , ");
                }

                Console.WriteLine(")");
            }
            #endregion
            */

            Console.ReadKey();
        }
    }
}

