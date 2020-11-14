using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CavakazeRenderer.IDGenerator
{
    /// <summary>
    /// Ordered ID Generator generated ID sequence is natural numbers, from 0 and next the value of number in current sequence increased 1.
    /// </summary>
    class OrderedIDGenerator: IDGenerator
    {
        private ulong ID_Idx;

        public OrderedIDGenerator()
        {
            ID_Idx = 0;
            isRecyclable = false;
        }

        public OrderedIDGenerator(bool _isRecyclable)
        {
            ID_Idx = 0;
            isRecyclable = _isRecyclable;
        }

        //ToDo: Implemented full AllocateID().
        public override ulong AllocateID()
        {
            ulong result;
            if (isRecyclable)
            {
                result = ID_Idx;
                ++ID_Idx;
            }
            else
            {
                result = ID_Idx;
                ++ID_Idx;
            }
            return result;
        }

        public override bool RecycleID(ulong ID)
        {
            return true;
        }
    }
}
