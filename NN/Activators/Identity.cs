using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Activators
{
    class Identity : IActivator
    {
        public double ActivatorSlope(double val)
        {
            return 1;
        }

        public double ActivatorValue(double val)
        {
            return val;
        }
    }
}
