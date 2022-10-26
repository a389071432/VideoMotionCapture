using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;

//Used for parse a selected BVH file
public class BVHParser
{
    public int nFrames = 0;
    public float frameInterval = 1000f / 60f;
    public BVHNode root;
    private List<BVHNode> nodes;

    static private char[] charMap = null;
    private float[][] channels;
    private string bvhText;
    private int pos = 0;

    public class BVHNode
    {
        public string name;
        public List<BVHNode> children;
        public float offsetX, offsetY, offsetZ;
        public int[] channelOrder;
        public int channelNumber;
        public BVHChannel[] channels;

        private BVHParser bp;
        public struct BVHChannel
        {
            public bool enabled;
            public float[] values;
        }

        public BVHNode(BVHParser parser, bool root,out bool success)
        {
            bp = parser;
            bp.nodes.Add(this);
            channels = new BVHChannel[6];
            channelOrder = new int[6] { 0, 1, 2, 5, 3, 4 };
            children = new List<BVHNode>();
            success = true;
            bp.skip();
            if (root)
            {
                if (!bp.Expect("ROOT"))
                {
                    success = false;
                    return;
                }
            }
            else
            {
                if (!bp.Expect("JOINT"))
                {
                    success = false;
                    return;
                }
            }
            if (!bp.getString(out name))
            {
                success = false;
                return;
            }
            bp.skip();
            if (!bp.Expect("{"))
            {
                success = false;
                return;
            }
            bp.skip();
            if (!bp.Expect("OFFSET"))
            {
                success = false;
                return;
            }
            bp.skip();

            if (!bp.getFloat(out offsetX))
            {
                success = false;
                return;
            }
            bp.skip();
            if (!bp.getFloat(out offsetY))
            {
                success = false;
                return;
            }
            bp.skip();
            if (!bp.getFloat(out offsetZ))
            {
                success = false;
                return;
            }
            bp.skip();
            if (!bp.Expect("CHANNELS"))
            {
                success = false;
                return;
            }

            bp.skip();
            if (!bp.getInt(out channelNumber))
            {
                success = false;
                return;
            }
            if (!(channelNumber >= 1 && channelNumber <= 6))
            {
                success = false;
                return;
            }

            for (int i = 0; i < channelNumber; i++)
            {
                bp.skip();
                int channelId;
                if (!bp.getChannel(out channelId))
                {
                    success = false;
                    return;
                }
                channelOrder[i] = channelId;
                channels[channelId].enabled = true;
            }

            char peek = ' ';
            do
            {
                float ignored;
                bp.skip();
                if (!bp.peek(out peek))
                {
                    success = false;
                    return;
                }
                switch (peek)
                {
                    case 'J':
                        bool success_temp = true;
                        BVHNode child = new BVHNode(bp, false,out success_temp);
                        if (!success_temp)
                        {
                            success = false;
                            return;
                        }
                        children.Add(child);
                        break;
                    case 'E':
                        if (!bp.Expect("End Site"))
                        {
                            success = false;
                            return;
                        }
                        bp.skip();
                        if (!bp.Expect("{"))
                        {
                            success = false;
                            return;
                        }
                        bp.skip();
                        if (!bp.Expect("OFFSET"))
                        {
                            success = false;
                            return;
                        }
                        bp.skip();
                        if (!bp.getFloat(out ignored))
                        {
                            success = false;
                            return;
                        }
                        bp.skip();
                        if (!bp.getFloat(out ignored))
                        {
                            success = false;
                            return;
                        }
                        bp.skip();
                        if (!bp.getFloat(out ignored))
                        {
                            success = false;
                            return;
                        }
                        bp.skip();
                        if (!bp.Expect("}"))
                        {
                            success = false;
                            return;
                        }
                        break;
                    case '}':
                        if (!bp.Expect("}"))
                        {
                            success = false;
                            return;
                        }
                        break;
                    default:
                        success = false;
                        return;
                        break;
                }
            } while (peek != '}');
        }
    }

    private bool peek(out char c)
    {
        c = ' ';
        if (pos >= bvhText.Length)
        {
            return false;
        }
        c = bvhText[pos];
        return true;
    }

    public bool Expect(string text)
    {
        foreach (char c in text)
        {
            if (pos >= bvhText.Length || (c != bvhText[pos] && bvhText[pos] < 256 && c != charMap[bvhText[pos]]))
            {
                return false;
            }
            pos++;
        }
        return true;
    }

    private bool getString(out string text)
    {
        text = "";
        while (pos < bvhText.Length && bvhText[pos] != '\n' && bvhText[pos] != '\r')
        {
            text += bvhText[pos++];
        }
        text = text.Trim();

        return (text.Length != 0);
    }

    private bool getChannel(out int channel)
    {
        channel = -1;
        if (pos + 1 >= bvhText.Length)
        {
            return false;
        }
        switch (bvhText[pos])
        {
            case 'x':
            case 'X':
                channel = 0;
                break;
            case 'y':
            case 'Y':
                channel = 1;
                break;
            case 'z':
            case 'Z':
                channel = 2;
                break;
            default:
                return false;
        }
        pos++;
        switch (bvhText[pos])
        {
            case 'p':
            case 'P':
                pos++;
                return Expect("osition");
            case 'r':
            case 'R':
                pos++;
                channel += 3;
                return Expect("otation");
            default:
                return false;
        }
    }

    private bool getInt(out int v)
    {
        bool negate = false;
        bool digitFound = false;
        v = 0;

        if (pos < bvhText.Length && bvhText[pos] == '-')
        {
            negate = true;
            pos++;
        }
        else if (pos < bvhText.Length && bvhText[pos] == '+')
        {
            pos++;
        }

        while (pos < bvhText.Length && bvhText[pos] >= '0' && bvhText[pos] <= '9')
        {
            v = v * 10 + (int)(bvhText[pos++] - '0');
            digitFound = true;
        }

        if (negate)
        {
            v *= -1;
        }
        if (!digitFound)
        {
            v = -1;
        }
        return digitFound;
    }

    private bool getFloat(out float v)
    {
        bool negate = false;
        bool digitFound = false;
        int i = 0;
        v = 0f;

        if (pos < bvhText.Length && bvhText[pos] == '-')
        {
            negate = true;
            pos++;
        }
        else if (pos < bvhText.Length && bvhText[pos] == '+')
        {
            pos++;
        }

        while (pos < bvhText.Length && bvhText[pos] >= '0' && bvhText[pos] <= '9')
        {
            v = v * 10 + (float)(bvhText[pos++] - '0');
            digitFound = true;
        }

        if (pos < bvhText.Length && (bvhText[pos] == '.' || bvhText[pos] == ','))
        {
            pos++;

            float fac = 0.1f;
            while (pos < bvhText.Length && bvhText[pos] >= '0' && bvhText[pos] <= '9' && i < 128)
            {
                v += fac * (float)(bvhText[pos++] - '0');
                fac *= 0.1f;
                digitFound = true;
            }
        }

        if (negate)
        {
            v *= -1f;
        }
        if (!digitFound)
        {
            v = float.NaN;
        }

        if (digitFound && pos < bvhText.Length && (bvhText[pos] == 'e' || bvhText[pos] == 'E'))
        {
            pos++;

            int scientificSign = 1;
            if (pos < bvhText.Length && bvhText[pos] == '-')
            {
                scientificSign = -1;
                pos++;
            }
            else if (pos < bvhText.Length && bvhText[pos] == '+')
            {
                pos++;
            }

            int power = 0;
            while (pos < bvhText.Length && bvhText[pos] >= '0' && bvhText[pos] <= '9')
            {
                power = power * 10 + (bvhText[pos++] - '0');
            }

            power *= scientificSign;

            v *= (float)Math.Pow(10, power);
        }

        return digitFound;
    }

    private void skip()
    {
        while (pos < bvhText.Length && (bvhText[pos] == ' ' || bvhText[pos] == '\t' || bvhText[pos] == '\n' || bvhText[pos] == '\r'))
        {
            pos++;
        }
    }

    private void skipInLine()
    {
        while (pos < bvhText.Length && (bvhText[pos] == ' ' || bvhText[pos] == '\t'))
        {
            pos++;
        }
    }

    private bool newLine()
    {
        bool foundNewline = false;
        skipInLine();
        while (pos < bvhText.Length && (bvhText[pos] == '\n' || bvhText[pos] == '\r'))
        {
            foundNewline = true;
            pos++;
        }
        return foundNewline;
    }

    private bool parse(bool overrideFrameTime, float time)
    {
        if (charMap == null)
        {
            charMap = new char[256];
            for (int i = 0; i < 256; i++)
            {
                if (i >= 'a' && i <= 'z')
                {
                    charMap[i] = (char)(i - 'a' + 'A');
                }
                else if (i == '\t' || i == '\n' || i == '\r')
                {
                    charMap[i] = ' ';
                }
                else
                {
                    charMap[i] = (char)i;
                }
            }
        }

        skip();

        if (!Expect("HIERARCHY"))
            return false;
        nodes = new List<BVHNode>();
        bool success;
        root = new BVHNode(this, true,out success);
        if (!success)
            return false;
        skip();
        if (!Expect("MOTION"))
            return false;
        skip();
        if (!Expect("FRAMES:"))
            return false;
        skip();
        if (!getInt(out nFrames))
            return false;
        skip();
        if (!Expect("FRAME TIME:"))
            return false;
        skip();
        if (!getFloat(out frameInterval))
            return false;

        if (overrideFrameTime)
        {
            frameInterval = time;
        }

        int totalChannels = 0;
        foreach (BVHNode bone in nodes)
        {
            totalChannels += bone.channelNumber;
        }
        int channel = 0;
        channels = new float[totalChannels][];
        foreach (BVHNode bone in nodes)
        {
            for (int i = 0; i < bone.channelNumber; i++)
            {
                channels[channel] = new float[nFrames];
                bone.channels[bone.channelOrder[i]].values = channels[channel++];
            }
        }

        for (int i = 0; i < nFrames; i++)
        {
            if (!newLine())
                return false;
            for (channel = 0; channel < totalChannels; channel++)
            {
                skipInLine();
                if (!getFloat(out channels[channel][i]))
                    return false;
            }
        }
        return true;
    }

    public bool parseFile(string bvhText)
    {
        pos = 0;
        this.bvhText = bvhText;
        return parse(false, 0f);
    }
}
