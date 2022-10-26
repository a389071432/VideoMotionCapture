using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;

//Customized data structure for 3-dimensinal matrix as Unity does not provide it
public class Matrix3D
{
    public float[,] m;
    public Vector3 vec0;
    public Vector3 vec1;
    public Vector3 vec2;
    public Matrix3D(float[,] a)
    {
        m = new float[3, 3];
        m[0, 0] = a[0, 0];
        m[1, 0] = a[1, 0];
        m[2, 0] = a[2, 0];

        m[0, 1] = a[0, 1];
        m[1, 1] = a[1, 1];
        m[2, 1] = a[2, 1];

        m[0, 2] = a[0, 2];
        m[1, 2] = a[1, 2];
        m[2, 2] = a[2, 2];

        vec0 = new Vector3(m[0, 0], m[1, 0], m[2, 0]);
        vec1 = new Vector3(m[0, 1], m[1, 1], m[2, 1]);
        vec2 = new Vector3(m[0, 2], m[1, 2], m[2, 2]);
    }

    public Matrix3D(Vector3 _vec0, Vector3 _vec1, Vector3 _vec2)
    {
        m = new float[3, 3];
        vec0 = _vec0;
        vec1 = _vec1;
        vec2 = _vec2;

        m[0, 0] = vec0.x;
        m[1, 0] = vec0.y;
        m[2, 0] = vec0.z;

        m[0, 1] = vec1.x;
        m[1, 1] = vec1.y;
        m[2, 1] = vec1.z;

        m[0, 2] = vec2.x;
        m[1, 2] = vec2.y;
        m[2, 2] = vec2.z;
    }

    public static Matrix3D operator *(Matrix3D a, Matrix3D b)
    {
        Vector3 v0 = a.vec0 * b.m[0, 0] + a.vec1 * b.m[1, 0] + a.vec2 * b.m[2, 0];
        Vector3 v1 = a.vec0 * b.m[0, 1] + a.vec1 * b.m[1, 1] + a.vec2 * b.m[2, 1];
        Vector3 v2 = a.vec0 * b.m[0, 2] + a.vec1 * b.m[1, 2] + a.vec2 * b.m[2, 2];
        return new Matrix3D(v0, v1, v2);
    }
}

//Responsible for mapping data in a BVH file to the Avatar
//Providing the APIs for animation controlling (used by AvatarDriver.cs)
public class AnimationHandler
{
    public Animator targetAvatar;
    public string clipName;
    public Animation anim;
    public AnimationClip clip;   
    private float rate = 1;
    static private int clipCount = 0;
    private BVHParser bp;
    private string pathPrefix;
    private Transform rootBone;
    private int frames;
    private Dictionary<string, string> nameMap;
    private Dictionary<Transform, Quaternion> initRotation;  //init rotation of bones
    private Dictionary<Transform, BVHParser.BVHNode> bone2node; //mapping bone of avatar to node in BVH 
    private Dictionary<Transform, string> bone2path;    //path of each bone, used for creating animaion clip
    private Dictionary<string, int> name2index;
    private Matrix3D[] T;
    private Matrix3D[] invT;

    //convert euler angles in BVH to Quaternion
    //intrinsic euler angles need to be converted to extrinsic angles
    private Quaternion getLocalRotation(Vector3 euler, string name)
    {
        float x = euler.x * Mathf.Deg2Rad;
        float y = euler.y * Mathf.Deg2Rad;
        float z = euler.z * Mathf.Deg2Rad;
        //Debug.Log(euler);
        float[,] dcm0 = new float[3, 3];
        dcm0[0, 0] = Mathf.Cos(z) * Mathf.Cos(y) - Mathf.Sin(z) * Mathf.Sin(x) * Mathf.Sin(y);
        dcm0[1, 0] = Mathf.Sin(z) * Mathf.Cos(y) + Mathf.Cos(z) * Mathf.Sin(x) * Mathf.Sin(y);
        dcm0[2, 0] = -Mathf.Cos(x) * Mathf.Sin(y);
        dcm0[0, 1] = -Mathf.Sin(z) * Mathf.Cos(x);
        dcm0[1, 1] = Mathf.Cos(z) * Mathf.Cos(x);
        dcm0[2, 1] = Mathf.Sin(x);
        dcm0[0, 2] = Mathf.Cos(z) * Mathf.Sin(y) + Mathf.Sin(z) * Mathf.Sin(x) * Mathf.Cos(y);
        dcm0[1, 2] = Mathf.Sin(z) * Mathf.Sin(y) - Mathf.Cos(z) * Mathf.Sin(x) * Mathf.Cos(y);
        dcm0[2, 2] = Mathf.Cos(x) * Mathf.Cos(y);

        Matrix3D dcm1 = (new Matrix3D(new Vector3(1, 0, 0), new Vector3(0, 0, 1), new Vector3(0, 1, 0))) * (new Matrix3D(dcm0)) * (new Matrix3D(new Vector3(1, 0, 0), new Vector3(0, 0, 1), new Vector3(0, 1, 0)));
        Matrix3D dcm2 = invT[name2index[name]] * dcm1 * T[name2index[name]];
        float angleZ = Mathf.Atan2(dcm2.m[1, 0], dcm2.m[1, 1]) * Mathf.Rad2Deg;
        float angleX = Mathf.Asin(-dcm2.m[1, 2]) * Mathf.Rad2Deg;
        float angleY = Mathf.Atan2(dcm2.m[0, 2], dcm2.m[2, 2]) * Mathf.Rad2Deg;

        return Quaternion.Euler(angleX, angleY, angleZ);
    }

    //Set curves for the animation clip recursively
    private void getCurves(Transform bone)
    {
        if (!nameMap.ContainsKey(bone.name))
            return;

        bool posX = false;
        bool posY = false;
        bool posZ = false;
        bool rotX = false;
        bool rotY = false;
        bool rotZ = false;

        float[][] values = new float[6][];
        Keyframe[][] keyframes = new Keyframe[7][];
        string[] props = new string[7];

        string path = bone2path[bone];
        BVHParser.BVHNode node = bone2node[bone];

        // This needs to be changed to gather from all channels into two vector3, invert the coordinate system transformation and then make keyframes from it
        for (int channel = 0; channel < 6; channel++)
        {
            if (!node.channels[channel].enabled)
            {
                continue;
            }

            switch (channel)
            {
                case 0:
                    posX = true;
                    props[channel] = "localPosition.x";
                    break;
                case 1:
                    posY = true;
                    props[channel] = "localPosition.y";
                    break;
                case 2:
                    posZ = true;
                    props[channel] = "localPosition.z";
                    break;
                case 3:
                    rotX = true;
                    props[channel] = "localRotation.x";
                    break;
                case 4:
                    rotY = true;
                    props[channel] = "localRotation.y";
                    break;
                case 5:
                    rotZ = true;
                    props[channel] = "localRotation.z";
                    break;
                default:
                    channel = -1;
                    break;
            }
            if (channel == -1)
            {
                continue;
            }

            keyframes[channel] = new Keyframe[frames];
            values[channel] = node.channels[channel].values;
            if (rotX && rotY && rotZ && keyframes[6] == null)
            {
                keyframes[6] = new Keyframe[frames];
                props[6] = "localRotation.w";
            }
        }

        float time = 0f;
        if (posX && posY && posZ)
        {
            Vector3 offset;
            offset = new Vector3(node.offsetX, node.offsetZ, node.offsetY);
            for (int i = 0; i < frames; i++)
            {
                time += bp.frameInterval;
                keyframes[0][i].time = time;
                keyframes[1][i].time = time;
                keyframes[2][i].time = time;
                keyframes[0][i].value = values[0][i];
                keyframes[1][i].value = values[1][i];
                keyframes[2][i].value = values[2][i];
                if (bone == rootBone)
                {
                    keyframes[0][i].value = 1.500806e-09f;
                    keyframes[1][i].value = 0.0004262662f;
                    keyframes[2][i].value = 0.007740487f;
                }
            }
            if (bone == rootBone)
            {
                clip.SetCurve(path, typeof(Transform), props[0], new AnimationCurve(keyframes[0]));
                clip.SetCurve(path, typeof(Transform), props[1], new AnimationCurve(keyframes[1]));
                clip.SetCurve(path, typeof(Transform), props[2], new AnimationCurve(keyframes[2]));
            }
        }

        time = 0f;
        if (rotX && rotY && rotZ)
        {
            //Quaternion oldRotation = bone.transform.rotation;
            for (int i = 0; i < frames; i++)
            {
                Vector3 eulerBVH = new Vector3(values[3][i], values[4][i], values[5][i]);
                Quaternion rot = initRotation[bone] * getLocalRotation(eulerBVH, node.name);
                keyframes[3][i].value = rot.x;
                keyframes[4][i].value = rot.y;
                keyframes[5][i].value = rot.z;
                keyframes[6][i].value = rot.w;
                time += bp.frameInterval;
                keyframes[3][i].time = time;
                keyframes[4][i].time = time;
                keyframes[5][i].time = time;
                keyframes[6][i].time = time;
            }
            //bone.transform.rotation = oldRotation;
            clip.SetCurve(path, typeof(Transform), props[3], new AnimationCurve(keyframes[3]));
            clip.SetCurve(path, typeof(Transform), props[4], new AnimationCurve(keyframes[4]));
            clip.SetCurve(path, typeof(Transform), props[5], new AnimationCurve(keyframes[5]));
            clip.SetCurve(path, typeof(Transform), props[6], new AnimationCurve(keyframes[6]));
        }

        for (int i = 0; i < bone.childCount; i++)
            getCurves(bone.GetChild(i));
    }

    private void getPathAndInitRotation(Transform current, string path)
    {
        bone2path.Add(current, path);
        Quaternion R = new Quaternion(current.localRotation.x, current.localRotation.y, current.localRotation.z, current.localRotation.w);
        initRotation.Add(current, R);
        for (int i = 0; i < current.childCount; i++)
            getPathAndInitRotation(current.GetChild(i), path + "/" + current.GetChild(i).name);
    }

    //Assign corresponding node for each bone
    private void getMapping(BVHParser.BVHNode node, Transform bone)
    {
        bone2node.Add(bone, node);
        for (int i = 0; i < bone.childCount; i++)
        {
            Transform childBone = bone.GetChild(i);
            foreach (BVHParser.BVHNode childNode in node.children)
            {
                if (childNode.name == nameMap[childBone.name])
                {
                    getMapping(childNode, childBone);
                    break;
                }
            }
        }
    }

    //Load the selected BVH file
    public void Load()
    {
        if (bp == null)
        {
            throw new InvalidOperationException("No BVH file has been parsed.");
        }
        frames = bp.nFrames;
        clip = new AnimationClip();
        clip.name = "BVHClip (" + (clipCount++) + ")";
        if (clipName != "")
        {
            clip.name = clipName;
        }
        clip.legacy = true;
        Vector3 targetAvatarPosition = targetAvatar.transform.position;
        Quaternion targetAvatarRotation = targetAvatar.transform.rotation;
        targetAvatar.transform.position = new Vector3(0.0f, 0.0f, 0.0f);
        targetAvatar.transform.rotation = Quaternion.identity;

        bone2node = new Dictionary<Transform, BVHParser.BVHNode>();
        getMapping(bp.root, rootBone);
        getCurves(rootBone);

        targetAvatar.transform.position = targetAvatarPosition;
        targetAvatar.transform.rotation = targetAvatarRotation;

        clip.EnsureQuaternionContinuity();
        if (anim == null)
        {
            anim = targetAvatar.gameObject.GetComponent<Animation>();
            if (anim == null)
            {
                anim = targetAvatar.gameObject.AddComponent<Animation>();
            }
        }
        anim.AddClip(clip, clip.name);
        anim.clip = clip;
    }

    public bool parseFile(string filePath)
    {
        StreamReader sr = new StreamReader(filePath);
        if (!bp.parseFile(sr.ReadToEnd()))
            return false;
        sr.Close();
        return true;
    }

    public void setAvatar(Animator avatar, string _pathPrefix, Transform _rootBone)
    {
        targetAvatar = avatar;
        pathPrefix = _pathPrefix;
        rootBone = _rootBone;
        initForAvatar();
    }

    //APIs for controlling the play of the animation
    public void Play()
    {
        if (clip is null)
            return;
        anim[clip.name].speed = rate;
        anim.Play(clip.name);
    }

    public void Reset()
    {
        if (clip is null)
            return;
        anim[clip.name].time = 0;
        anim.Stop();
        resetPose(rootBone);
    }

    public void Pause()
    {
        if (clip is null)
            return;
        anim[clip.name].speed = 0;
    }

    public void Continue()
    {
        if (clip is null)
            return;
        anim[clip.name].speed = rate;
    }

    public void setToTime(float value)
    {
        if (clip is null)
            return;
        if (this.isPlaying())
            return;
        anim[clip.name].time = value * bp.frameInterval * bp.nFrames;
    }

    public void setSpeed(float _rate)
    {
        rate = _rate;
        if (!isPlaying())
            return;
        anim[clip.name].speed = _rate;
    }

    public bool Ready()
    {
        return !(clip is null);
    }

    public bool isPlaying()
    {
        if (clip is null)
            return false;
        return anim[clip.name].speed > 0;
    }

    //Set Avatar to T-pose
    private void resetPose(Transform current)
    {
        Quaternion R = new Quaternion(initRotation[current].x, initRotation[current].y, initRotation[current].z, initRotation[current].w);
        current.localRotation = R;
        for (int i = 0; i < current.childCount; i++)
            resetPose(current.GetChild(i));
    }

    private void initForAvatar()
    {
        if (targetAvatar is null)
            return;
        bone2path = new Dictionary<Transform, string>();
        initRotation = new Dictionary<Transform, Quaternion>();
        getPathAndInitRotation(rootBone, pathPrefix + "/" + rootBone.name);

        nameMap = new Dictionary<string, string> { {"Hips","Hip"},
                                                     { "RightUpLeg","RightHip" },
                                                     { "RightLeg","RightKnee" },
                                                     { "RightFoot","RightAnkle" },
                                                     { "LeftUpLeg","LeftHip" },
                                                     { "LeftLeg","LeftKnee" },
                                                     { "LeftFoot","LeftAnkle" },
                                                     { "Spine","Spine" },
                                                     { "Spine1","Chest" },
                                                     { "Spine2","Thorax" },
                                                     { "Neck","Neck" },
                                                     { "Head","Head" },
                                                     { "LeftShoulder","LeftShoulder" },
                                                     { "LeftArm","LeftUpArm" },
                                                     { "LeftForeArm","LeftElbow" },
                                                     { "LeftHand","LeftWrist" },
                                                     { "RightShoulder","RightShoulder" },
                                                     { "RightArm","RightUpArm" },
                                                     { "RightForeArm","RightElbow" },
                                                     { "RightHand","RightWrist" }
                                                   };

        //T[] and invT are used for transformation of local coordinate for each bone,
        //used in function getLocalRotation() 
        T = new Matrix3D[20] {new Matrix3D(new Vector3(1,0,0),new Vector3(0,0.866f,0.5f),new Vector3(0,-0.5f,0.866f)),   //Hip
                              new Matrix3D(new Vector3(-1,0,0),new Vector3(0,-1,0),new Vector3(0,0,1)),
                              new Matrix3D(new Vector3(-1,0,0),new Vector3(0,-1,0),new Vector3(0,0,1)),
                              new Matrix3D(new Vector3(-1,0,0),new Vector3(0,-1,0),new Vector3(0,0,1)),  //RightAnkle
                              new Matrix3D(new Vector3(-1,0,0),new Vector3(0,-1,0),new Vector3(0,0,1)),
                              new Matrix3D(new Vector3(-1,0,0),new Vector3(0,-1,0),new Vector3(0,0,1)),
                              new Matrix3D(new Vector3(-1,0,0),new Vector3(0,-1,0),new Vector3(0,0,1)),  //LeftAnkle
                              new Matrix3D(new Vector3(1,0,0),new Vector3(0,1,0),new Vector3(0,0,1)),
                              new Matrix3D(new Vector3(1,0,0),new Vector3(0,1,0),new Vector3(0,0,1)),
                              new Matrix3D(new Vector3(1,0,0),new Vector3(0,1,0),new Vector3(0,0,1)),
                              new Matrix3D(new Vector3(1,0,0),new Vector3(0,1,0),new Vector3(0,0,1)),
                              new Matrix3D(new Vector3(1,0,0),new Vector3(0,1,0),new Vector3(0,0,1)),
                              new Matrix3D(new Vector3(0,0,1),new Vector3(-0.866f,-0.5f,0),new Vector3(0.5f,-0.866f,0)),   //LeftShoulder
                              new Matrix3D(new Vector3(0,0,1),new Vector3(-1,0,0),new Vector3(0,-1,0)),
                              new Matrix3D(new Vector3(0,0,1),new Vector3(-1,0,0),new Vector3(0,-1,0)),
                              new Matrix3D(new Vector3(0,0,1),new Vector3(-1,0,0),new Vector3(0,-1,0)),
                              new Matrix3D(new Vector3(0,0,-1),new Vector3(0.866f,-0.5f,0),new Vector3(-0.5f,-0.866f,0)),   //RightShoulder
                              new Matrix3D(new Vector3(0,0,-1),new Vector3(1,0,0),new Vector3(0,-1,0)),
                              new Matrix3D(new Vector3(0,0,-1),new Vector3(1,0,0),new Vector3(0,-1,0)),
                              new Matrix3D(new Vector3(0,0,-1),new Vector3(1,0,0),new Vector3(0,-1,0))

        };
        invT = new Matrix3D[20] {new Matrix3D(new Vector3(1,0,0),new Vector3(0,0.866f,-0.5f),new Vector3(0,0.5f,0.866f)),   //Hip
                                 new Matrix3D(new Vector3(-1,0,0),new Vector3(0,-1,0),new Vector3(0,0,1)),
                                 new Matrix3D(new Vector3(-1,0,0),new Vector3(0,-1,0),new Vector3(0,0,1)),
                                 new Matrix3D(new Vector3(-1,0,0),new Vector3(0,-1,0),new Vector3(0,0,1)),  //RightAnkle
                                 new Matrix3D(new Vector3(-1,0,0),new Vector3(0,-1,0),new Vector3(0,0,1)),
                                 new Matrix3D(new Vector3(-1,0,0),new Vector3(0,-1,0),new Vector3(0,0,1)),
                                 new Matrix3D(new Vector3(-1,0,0),new Vector3(0,-1,0),new Vector3(0,0,1)),  //LeftAnkle
                                 new Matrix3D(new Vector3(1,0,0),new Vector3(0,1,0),new Vector3(0,0,1)),
                                 new Matrix3D(new Vector3(1,0,0),new Vector3(0,1,0),new Vector3(0,0,1)),
                                 new Matrix3D(new Vector3(1,0,0),new Vector3(0,1,0),new Vector3(0,0,1)),
                                 new Matrix3D(new Vector3(1,0,0),new Vector3(0,1,0),new Vector3(0,0,1)),
                                 new Matrix3D(new Vector3(1,0,0),new Vector3(0,1,0),new Vector3(0,0,1)),
                                 new Matrix3D(new Vector3(0,-0.866f,0.5f),new Vector3(0,-0.5f,-0.866f),new Vector3(1,0,0)),  //leftShoulder
                                 new Matrix3D(new Vector3(0,-1,0),new Vector3(0,0,-1),new Vector3(1,0,0)),
                                 new Matrix3D(new Vector3(0,-1,0),new Vector3(0,0,-1),new Vector3(1,0,0)),
                                 new Matrix3D(new Vector3(0,-1,0),new Vector3(0,0,-1),new Vector3(1,0,0)),
                                 new Matrix3D(new Vector3(0,0.866f,-0.5f),new Vector3(0,-0.5f,-0.866f),new Vector3(-1,0,0)),  //RightShoulder
                                 new Matrix3D(new Vector3(0,1,0),new Vector3(0,0,-1),new Vector3(-1,0,0)),
                                 new Matrix3D(new Vector3(0,1,0),new Vector3(0,0,-1),new Vector3(-1,0,0)),
                                 new Matrix3D(new Vector3(0,1,0),new Vector3(0,0,-1),new Vector3(-1,0,0))
        };
    }

    public AnimationHandler()
    {
        bp = new BVHParser();
        name2index = new Dictionary<string, int>{ {"Hip",0},
                                                  {"RightHip",1},
                                                  {"RightKnee",2},
                                                  {"RightAnkle",3},
                                                  {"LeftHip",4},
                                                  {"LeftKnee",5},
                                                  {"LeftAnkle",6},
                                                  {"Spine",7},
                                                  {"Chest",8},
                                                  {"Thorax",9},
                                                  {"Neck",10},
                                                  {"Head",11},
                                                  {"LeftShoulder",12},
                                                  {"LeftUpArm",13},
                                                  {"LeftElbow",14},
                                                  {"LeftWrist",15},
                                                  {"RightShoulder",16},
                                                  {"RightUpArm",17},
                                                  {"RightElbow",18},
                                                  {"RightWrist",19}};
    }
}