# Deploying the Application on a Raspberry Pi with Cloudflare Tunnel

This guide explains how to securely expose a web application running on a Raspberry Pi to the internet using Cloudflare Tunnel. This method does not require a public IP address or complex firewall configurations.

## ✨ Key Advantages

*   **No Public IP Needed**: Your Raspberry Pi remains hidden from the public internet.
*   **Secure by Default**: Traffic is encrypted with HTTPS, and Cloudflare provides an extra layer of protection against DDoS attacks.
*   **Stable URL**: You get a permanent, publicly accessible URL for your application.
*   **Free**: Cloudflare's basic Tunnel service is free to use.

## 📋 Prerequisites

1.  **Raspberry Pi**: A Raspberry Pi 4 or newer is recommended, running a 64-bit version of Raspberry Pi OS.
2.  **Docker & Docker Compose**: Docker and Docker Compose must be installed on your Raspberry Pi.
3.  **Cloudflare Account**: A free account with Cloudflare is required.
4.  **(Optional) Domain Name**: If you own a domain name, you can manage it through Cloudflare to create a custom URL (e.g., `reconciliation.yourdomain.com`). If not, Cloudflare will provide a free, random `trycloudflare.com` URL.

---

## 🚀 Step-by-Step Deployment Guide

### Step 1: Prepare the Application on Your Raspberry Pi

1.  **Open a terminal** on your Raspberry Pi (either directly or via SSH).

2.  **Clone your project repository**:
    ```bash
    git clone <your-repo-url>
    cd accounting-reconciliation
    ```
    *(Replace `<your-repo-url>` with the actual URL of your Git repository)*

3.  **Build and Run the Application with Docker Compose**:
    This command will build the Docker image specifically for your Pi's ARM architecture and start the web service in the background.
    ```bash
    docker compose up -d --build
    ```

4.  **Verify the Application is Running Locally**:
    Open a web browser *on your Raspberry Pi* and navigate to `http://localhost:5000`. You should see the application's web interface. This confirms the container is working correctly.

### Step 2: Set Up Cloudflare Tunnel

1.  **Install `cloudflared` on the Raspberry Pi**:
    Follow the official Cloudflare instructions to download and install the `cloudflared` daemon. For a 64-bit Raspberry Pi OS, you will typically choose the **Debian** package for the **arm64** architecture.

    Example command:
    ```bash
    wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
    sudo dpkg -i cloudflared-linux-arm64.deb
    ```

2.  **Authenticate `cloudflared`**:
    This command will open a browser window. Log in to your Cloudflare account and select a domain you want to authorize (or just authorize the account if you don't have a domain).
    ```bash
    cloudflared tunnel login
    ```
    Once authorized, a certificate file will be saved in the `~/.cloudflared/` directory.

3.  **Create a Tunnel**:
    Give your tunnel a memorable name. This command registers the tunnel with Cloudflare and creates a credentials file for it.
    ```bash
    cloudflared tunnel create reconciliation-app
    ```
    Take note of the **Tunnel UUID** and the path to the credentials file (`.json`) shown in the output. You will need these for the next step.

### Step 3: Configure and Run the Tunnel

1.  **Create a Configuration File**:
    You need to tell `cloudflared` where to send incoming traffic. Create a configuration file in the `~/.cloudflared/` directory.
    ```bash
    nano ~/.cloudflared/config.yml
    ```

2.  **Add the following content to `config.yml`**:
    Replace `<Your-Tunnel-UUID>` with the UUID from the previous step.

    ```yaml
    tunnel: <Your-Tunnel-UUID>
    credentials-file: /home/pi/.cloudflared/<Your-Tunnel-UUID>.json

    ingress:
      # This rule forwards traffic to your local web application
      - hostname: reconciliation.your-domain.com # <-- IMPORTANT: Change this!
        service: http://localhost:5000

      # This is a catch-all rule that returns a 404 error for any other traffic.
      - service: http_status:404
    ```

    **Hostname Configuration:**
    *   **If you have a domain:** Replace `reconciliation.your-domain.com` with the subdomain and domain you want to use.
    *   **If you don't have a domain:** You can omit the `hostname` line. Cloudflare will assign a random `*.trycloudflare.com` URL when you run the tunnel.

3.  **Route DNS to Your Tunnel (Only if using your own domain)**:
    This command creates the necessary DNS records in your Cloudflare account to point your chosen hostname to the tunnel.
    ```bash
    cloudflared tunnel route dns reconciliation-app reconciliation.your-domain.com
    ```

4.  **Run the Tunnel as a Service**:
    Running `cloudflared` as a system service ensures it starts automatically when your Raspberry Pi boots up.

    ```bash
    # Install the service
    sudo cloudflared service install

    # Start the service
    sudo systemctl start cloudflared

    # (Optional) Check the status of the service
    sudo systemctl status cloudflared
    ```

---

## ✅ All Done!

Your application is now securely accessible from anywhere on the internet at the URL you configured (e.g., `https://reconciliation.your-domain.com`). Cloudflare handles the HTTPS certificate automatically.

To view the tunnel's logs, you can use:
```bash
journalctl -u cloudflared -f
```