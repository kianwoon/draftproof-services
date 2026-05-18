export function browserNotificationsSupported() {
  return typeof window !== 'undefined' && 'Notification' in window;
}

export function browserNotificationsAvailable() {
  return browserNotificationsSupported() && window.isSecureContext;
}

export async function requestBrowserNotificationPermission() {
  if (!browserNotificationsAvailable()) return 'unsupported';
  if (Notification.permission !== 'default') return Notification.permission;

  try {
    return await Notification.requestPermission();
  } catch {
    return Notification.permission;
  }
}

export function showBrowserNotification({ title, body, tag, url }) {
  if (!browserNotificationsAvailable() || Notification.permission !== 'granted') {
    return false;
  }

  try {
    const notification = new Notification(title, {
      body,
      tag,
      renotify: false,
    });

    notification.onclick = () => {
      window.focus();
      if (url) {
        window.location.assign(url);
      }
      notification.close();
    };
    return true;
  } catch {
    return false;
  }
}
