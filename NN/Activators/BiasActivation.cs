using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Activators
{
    public class BiasActivation : IActivator
    {
        public double ActivatorSlope(double val)
        {
            return 1;
        }

        public double ActivatorValue(double val)
        {
            return 1;
        }
    }
}
