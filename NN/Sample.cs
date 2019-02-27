using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN
{
    public class Sample
    {
        public List<double> Inputs { get; set; }
        public List<double> Results { get; set; }


        public static void SaveSamplesAsFile(List<Sample> data, string path)
        {
            var jsonString = JsonConvert.SerializeObject(data);

            if (File.Exists(path))
            {
                File.Delete(path);
            }


            // Create a file to write to.
            using (StreamWriter sw = File.CreateText(path))
            {
                sw.Write(jsonString);
            }
        }
        public static List<Sample> LoadSamplesFromFile(string path)
        {
            var data = File.ReadAllText(path);

            return JsonConvert.DeserializeObject<List<Sample>>(data);
        }
    }
}