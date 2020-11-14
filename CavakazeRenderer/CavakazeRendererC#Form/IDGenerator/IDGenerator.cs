using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CavakazeRenderer.IDGenerator
{
    /// <summary>
    /// ID Generator is used for generating an unique ID every time per calling within an IDGenerator instance.
    /// </summary>
    abstract class IDGenerator
    {
        protected bool isRecyclable;

        public IDGenerator(bool _isRecyclable)
        {
            isRecyclable = _isRecyclable;
        }
        public IDGenerator()
        {
            isRecyclable = false;
        }

        abstract public ulong AllocateID();
        abstract public bool RecycleID(ulong ID);

    }
}
