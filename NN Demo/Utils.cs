using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN_Demo
{
    public static class Utils
    {

        private static readonly Random _random = new Random();

        public static double GetRandomWeight()
        {
            return (_random.NextDouble() - 0.5d);
        }
    }
}
