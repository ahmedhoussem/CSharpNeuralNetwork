using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_Demo
{
    public class LeakyReLU : IActivator
    {
        public double ActivatorValue(double val)
        {
            return val >= 0 ? val : val * .01;
        }

        public double ActivatorSlope(double val)
        {
            return val >= 0 ? 1 : .01;
        }

    }
}
