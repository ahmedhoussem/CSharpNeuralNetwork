using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.ModelFileSave
{
    public class NNData
    {
        public List<int> Details { get; set; }
        public List<IActivator> Activators { get; set; }
        public double LearningRate { get; set; }
        public double Momentum { get; set; }
        public List<WeightLink> Weights { get; set; }
    }
}
