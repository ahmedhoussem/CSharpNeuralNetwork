using Newtonsoft.Json;
using NN.Activators;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN
{
    public class Neuron
    {
        //NN info
        [JsonIgnore]
        public NeuralNet NNref { get; set; }
        public int LayerIndex { get; set; }
        public int NeuronIndex { get; set; }

        //Previous Neurons Link
        public List<WeightLink> PreviousLinks { get; set; } = new List<WeightLink>();
        //Next Neurons Link
        public List<WeightLink> NextLinks { get; set; } = new List<WeightLink>();

        public double Input { get; set; }

        public double Error { get; set; }

        public double Gamma { get; set; }

        public double ActivatedOutput { get; set; }

        [JsonConverter(typeof(IActivatorConverter))]
        public IActivator Activation { get; set; }

        public double Output { get; set; }


        public void ComputeInputs()
        {
            double sum = 0d;
            for (int i = 0; i < PreviousLinks.Count; i++)
            {
                sum += PreviousLinks[i].Weight * PreviousLinks[i].Left.ActivatedOutput;
            }

            Input = sum;
        }

        public void ComputeActivatedOutput()
        {
            ActivatedOutput = Activation.ActivatorValue(Input);
        }


        public void ComputeErrorOutput(double Ideal, ref double OutputError)
        {
            //Error = actual - ideal
            Error = ActivatedOutput - Ideal;

            OutputError += Math.Pow(Error, 2);

        }


        public void ComputeGammaOutput()
        {
            //Error = actual - ideal
            Gamma = Error * Activation.ActivatorSlope(ActivatedOutput);

        }

        public void ComputeDeltaWeightsOutput()
        {
            for (int i = 0; i < PreviousLinks.Count; i++)
            {
                PreviousLinks[i].AdjustDeltaOutput();
            }
        }

        public void ComputeGammaHidden()
        {
            Gamma = 0;

            for (int i = 0; i < NextLinks.Count; i++)
            {
                Gamma += NextLinks[i].Right.Gamma * NextLinks[i].Weight;
            }

            Gamma = Gamma * Activation.ActivatorSlope(Output);

            for (int i = 0; i < PreviousLinks.Count; i++)
            {
                PreviousLinks[i].Delta = PreviousLinks[i].Right.Gamma * PreviousLinks[i].Left.ActivatedOutput;
            }

        }

        public override string ToString()
        {
            var part1 = $"Layer : {LayerIndex} , Index : {NeuronIndex} , Activated output : { ActivatedOutput }, Gamma : { Gamma } ";
            part1 += $" \n Previous links : {PreviousLinks.Count}\n";
            foreach (var prev in PreviousLinks)
            {
                part1 += " " + prev.Weight + " ,";
            }
            part1 += $" \n Next links : {NextLinks.Count}\n";

            foreach (var next in NextLinks)
            {
                part1 += " " + next.Weight + " ,";
            }

            return part1;
        }


    }
}