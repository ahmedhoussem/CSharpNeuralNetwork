using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN
{
    public class ReLU : IActivator
    {
        public double ActivatorValue(double val)
        {
            return val >= 0 ? val : 0;
        }

        public double ActivatorSlope(double val)
        {
            return val >= 0 ? 1 : 0;
        }
    }
}
