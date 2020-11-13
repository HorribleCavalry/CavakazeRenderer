using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.Windows.Forms;

using CavakazeRenderer.CudaAPI;

namespace CavakazeRenderer
{
    public partial class CavakazeRendererMainForm : Form
    {
        Thread startRenderingThread;
        ThreadStart renderingStart;
        public CavakazeRendererMainForm()
        {
            InitializeComponent();
            RenderManager.OpenDebugConsole();
            renderingStart = new ThreadStart(RenderManager.StartRendering);
            startRenderingThread = new Thread(renderingStart);
        }

        private void flowLayoutPanel1_Paint(object sender, PaintEventArgs e)
        {

        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void label2_Click(object sender, EventArgs e)
        {

        }

        private void treeView1_AfterSelect(object sender, TreeViewEventArgs e)
        {

        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void label4_Click(object sender, EventArgs e)
        {

        }

        private void label6_Click(object sender, EventArgs e)
        {

        }

        private void AddButton_Click(object sender, EventArgs e)
        {
            Hierarchy.SelectedNode.Nodes.Add("A");
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Hierarchy.SelectedNode.Remove();
        }

        private void RenderButton_Click(object sender, EventArgs e)
        {
            switch (startRenderingThread.ThreadState)
            {
                case ThreadState.Running:
                    break;
                case ThreadState.StopRequested:
                    break;
                case ThreadState.SuspendRequested:
                    break;
                case ThreadState.Background:
                    break;
                case ThreadState.Unstarted:
                    startRenderingThread.Start();
                    break;
                case ThreadState.Stopped:
                    break;
                case ThreadState.WaitSleepJoin:
                    break;
                case ThreadState.Suspended:
                    break;
                case ThreadState.AbortRequested:
                    break;
                case ThreadState.Aborted:
                    break;
                default:
                    break;
            }
        }
    }
}
