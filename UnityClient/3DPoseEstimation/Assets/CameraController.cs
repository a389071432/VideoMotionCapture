using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraController : MonoBehaviour
{
    private Transform camTransform;
    private GameObject avatarCenter;
    private float cr = 2;
    private float cs = 100;
    private bool enable=true;
    // Start is called before the first frame update
    void Start()
    {
        avatarCenter = GameObject.Find("Hips");
        camTransform= GameObject.Find("MainCamera").transform;
    }

    // Update is called once per frame
    void Update()
    {
        if(enable)
          handleInput();
        camTransform.LookAt(avatarCenter.transform);
    }

    void handleInput()
    {
        if (Input.GetMouseButton(1))
        {
            float dx = Input.GetAxis("Mouse X");
            float dy = Input.GetAxis("Mouse Y");

            float r = cr*Mathf.Sqrt(dx * dx + dy * dy);
            Vector3 left = Vector3.Cross(camTransform.up, camTransform.forward);
            Vector3 axis = dx * camTransform.up + (-dy) * left;
            Vector3 nOFFSET = Quaternion.AngleAxis(r, axis) * (camTransform.position - avatarCenter.transform.position);
            camTransform.position = avatarCenter.transform.position + nOFFSET;
        }
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        if (scroll != 0)
        {
            Vector3 OFFSET = camTransform.position - avatarCenter.transform.position;
            Vector3 n = Vector3.Normalize(OFFSET);
            camTransform.position = avatarCenter.transform.position + OFFSET-n*cs*scroll;
        }
    }

    public void setEnable(bool _enable)
    {
        enable = _enable;
    }

    public Vector3 getPosition()
    {
        return camTransform.position;
    }

    public void setPosition(Vector3 _pos)
    {
        camTransform.position = _pos;
    }
    
    
}
