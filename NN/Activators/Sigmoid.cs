using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Activators
{
    public class Sigmoid : IActivator
    {
        public double ActivatorSlope(double val)
        {
            return val * (1.0 - val);
        }

        public double ActivatorValue(double val)
        {
            return 1 / (1 + Math.Exp(-val));
        }
    }
}
