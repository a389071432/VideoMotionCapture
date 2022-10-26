using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using UnityEditor;
using UnityEngine.Networking;

//Providing interactions in the Main interface for users 
public class FileBrowser : MonoBehaviour
{
    private string filePath;
    private string backendUrl;
    private BvhHandler bvhHandler;
    private GameObject panelMain;
    private GameObject panelSettings;
    private GameObject panelVisual;
    public GameObject textTip;
    private GameObject textTipMain;
    private GameObject camController;
    private float frameRate;
    private float factor;
    private int numInter;
    private string frameRate_str;
    private string factor_str;
    private string numInter_str;
    private string savePath;
    private int tipCnt = 0;
    public GameObject buttonVisual;
    Vector3[] poses;
    private int cnt;   //count for moving camera animation

    void Start()
    {
        //backendUrl = "http://38.22.105.160:8080/processVideo";
        backendUrl = "http://127.0.0.1:8080/processVideo";
        bvhHandler = new BvhHandler();
        camController = GameObject.Find("CameraController");
        panelMain = GameObject.Find("panelMain");
        panelSettings = GameObject.Find("panelSettings");
        panelVisual= GameObject.Find("panelVisual");
        textTipMain = GameObject.Find("textTipMain");
        panelMain.SetActive(true);
        panelSettings.SetActive(false);
        buttonVisual.SetActive(false);
        GameObject.Find("CameraController").GetComponent<CameraController>().setEnable(false);
        cnt = 0;
    }

    // Update is called once per frame
    void Update()
    {

    }

    //Callback function for inputFields
    public void onFrameRateEndEdit(string value)
    {
        frameRate_str = value;
    }

    public void onFactorEndEdit(string value)
    {
        factor_str = value;
    }

    public void onInterEndEdit(string value)
    {
        numInter_str = value;
    }

    public void onSavePath()
    {
        savePath = EditorUtility.SaveFilePanel("select a path to save", "", "","bvh");
        GameObject.Find("inputPath").GetComponent<InputField>().text = savePath;
    }

    private void onDataReceived()
    {
        textTipMain.GetComponent<Text>().text = "Motion extraction done!\n several settings is needed to create a BVH file :";
        textTipMain.SetActive(true);
        panelMain.SetActive(false);
        panelSettings.SetActive(true);
    }

    private bool checkInput()
    {
        if (!float.TryParse(frameRate_str, out frameRate))
            return false;
        if (frameRate < 0)
            return false;
        if (!float.TryParse(factor_str, out factor))
            return false;
        if (factor < 0 || factor > 1)
            return false;
        if (!int.TryParse(numInter_str, out numInter))
            return false;
        if (savePath.Length == 0)
            return false;
        return true;
    }

    //Callback function for submitting the settings for creating a BVH file
    public void onSettingsSubmit()
    {
        if (!checkInput())
        {
            textTipMain.GetComponent<Text>().text = "Incompleted or incorrect input !";
            return;
        }
        textTipMain.GetComponent<Text>().text = "Createing BVH file, please wait for seconds ......";
        bvhHandler.createBVH(poses, frameRate, factor, numInter);
        //bvhHandler.createBVH(poses, 24, 0, 0);
        bvhHandler.saveBVH(savePath);
        //bvhHandler.saveBVH("D:/bvh.bvh");
        textTipMain.GetComponent<Text>().text = "BVH file saved successfully!";
        panelSettings.SetActive(false);
        panelMain.SetActive(true);
    }

    //Callback function for buttons
    public void OpenFolder()
    {
        filePath = EditorUtility.OpenFilePanel("select a video", "D:/videopose3dinference/outputs", "mp4");
    }

    public void sendToBackend()
    {
        StartCoroutine(postVideo());        
    }

    //Transition animation
    public void moveCamera()
    {
        camController.GetComponent<CameraController>().setPosition(camController.GetComponent<CameraController>().getPosition()+new Vector3(0,0,1.8f));
        cnt += 1;
        if (cnt == 180)
        {
            CancelInvoke("moveCamera");
            textTip.SetActive(true);
            panelVisual.SetActive(true);
            buttonVisual.SetActive(true);
            camController.GetComponent<CameraController>().setEnable(true);
            cnt = 0;
        }
    }

    //Switching to Visualization Interface
    public void toVisual()
    {
        panelMain.SetActive(false);
        textTipMain.SetActive(false);
        GameObject.Find("Curtain").GetComponent<MeshRenderer>().material.SetColor("_Color", new Color(151/255, 179/255, 253/255, 0));
        InvokeRepeating("moveCamera", 0, 0.006f);
    }

    public void hideTipMain()
    {
        if (tipCnt == 0)
            textTipMain.SetActive(false);
        tipCnt -= 1;
    }

    //Send the selected video to backend for 3D pose estimation
    IEnumerator postVideo()
    {
        textTipMain.GetComponent<Text>().text = "Extracting motion from video.\n Please wait for a minute ...";
        WWWForm form = new WWWForm();
        form.AddBinaryData("video", File.ReadAllBytes(filePath));
        UnityWebRequest www = UnityWebRequest.Post(backendUrl, form);
        yield return www.SendWebRequest();
        if (www.result == UnityWebRequest.Result.ProtocolError || www.result == UnityWebRequest.Result.ConnectionError)
        {
            tipCnt += 1;
            textTipMain.GetComponent<Text>().text = "Network Error.\n Please retry.";
            Invoke("hideTipMain", 3.0f);
        }
        else
        {
            string ans = www.downloadHandler.text;
            string[] data = ans.Split(",");
            poses = new Vector3[(int)(data.Length/3)];
            int cnt = 0;
            for (int i = 0; i < data.Length; i += 3)
            {
                float x = float.Parse(data[i]);
                float y = float.Parse(data[i + 1]);
                float z = float.Parse(data[i + 2]);
                Vector3 dir = new Vector3(x, y, z);
                poses[cnt] = dir*10;
                cnt += 1;
            }
            onDataReceived();
        }
        www.Dispose();
    }
}
