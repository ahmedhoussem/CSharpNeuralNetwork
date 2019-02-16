using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_Demo
{
    public class WeightLink
    {
        [JsonIgnore]
        public Neuron Left { get; set; }

        [JsonIgnore]
        public Neuron Right { get; set; }

        public double Weight { get; set; }
        public double Delta { get; set; }

        public double PreviousChange { get; set; }

        public void RandomizeWeight(double amp)
        {
            Weight = Utils.GetRandomWeight(amp);
        }

        public void AdjustDeltaOutput()
        {
            Delta = Left.ActivatedOutput * Right.Gamma;
        }


        public void AdjustWeight(double LearningRate , double Momentum)
        {
            PreviousChange = (Delta * LearningRate) + (Momentum * PreviousChange); 
            Weight -= PreviousChange;
        }
    }
}
