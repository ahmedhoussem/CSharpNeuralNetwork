using Newtonsoft.Json;
using NN_Demo.Activators;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_Demo
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
            foreach( var link in PreviousLinks)
            {
                sum += link.Weight * link.Left.ActivatedOutput;
            }

            Input = sum;
        }

        public void ComputeActivatedOutput()
        {
            ActivatedOutput = Activation.ActivatorValue(Input);
        }


        public void ComputeErrorOutput(double Ideal , ref double OutputError)
        {
            //Error = actual - ideal
            Error =  ActivatedOutput - Ideal;

            OutputError += Math.Pow(Error , 2);

        }


        public void ComputeGammaOutput()
        {
            //Error = actual - ideal
            Gamma = Error * Activation.ActivatorSlope(ActivatedOutput);

        }

        public void ComputeDeltaWeightsOutput()
        {
            foreach(var link in PreviousLinks)
            {
                link.AdjustDeltaOutput();             
            }
        }

        public void ComputeGammaHidden()
        {
            Gamma = 0;

            foreach (var link in NextLinks)
            {
                Gamma += link.Right.Gamma * link.Weight;
            }

            Gamma = Gamma * Activation.ActivatorSlope(Output);

            foreach (var link in PreviousLinks)
            {
                link.Delta = link.Right.Gamma * link.Left.ActivatedOutput;
            }

        }

        public override string ToString()
        {
            var part1 = $"Layer : {LayerIndex} , Index : {NeuronIndex} , Activated output : { ActivatedOutput }, Gamma : { Gamma } ";
            part1 += $" \n Previous links : {PreviousLinks.Count}\n";
            foreach (var prev in PreviousLinks)
            {
                part1 += " " + prev.Weight +" ," ;
            }
            part1 += $" \n Next links : {NextLinks.Count}\n";

            foreach (var next in NextLinks)
            {
                part1 += " " + next.Weight + " ,";
            }

            return  part1;
        }


    }

}
