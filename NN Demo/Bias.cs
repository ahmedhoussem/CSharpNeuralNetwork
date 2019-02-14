using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_Demo
{
    public class Bias
    {
        public double ActivatedOutput { get; set; }
        public IActivator Activator { get; set; }

        public List<WeightLink> NextLinks { get; set; } = new List<WeightLink>();




    }
}
