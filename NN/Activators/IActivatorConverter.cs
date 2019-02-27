using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Activators
{
    public class IActivatorConverter : JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return (objectType == typeof(IActivator));
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {
            if (existingValue as Sigmoid != null)
                return serializer.Deserialize(reader, typeof(Sigmoid));

            return serializer.Deserialize(reader, typeof(BiasActivation));

        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            if (value as Sigmoid != null)
                serializer.Serialize(writer, value, typeof(Sigmoid));

            serializer.Serialize(writer, value, typeof(BiasActivation));
        }
    }
}