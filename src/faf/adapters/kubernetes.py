from datetime import datetime
from typing import List, Any, Dict

from faf.adapters.base import FailureAdapter
from faf.models import RunEvent, FailedStepInfo
from faf.exceptions import MissingDependenciesError

class KubernetesAdapter(FailureAdapter):
    @property
    def source_name(self) -> str:
        return "kubernetes"

    @property
    def available_signals(self) -> List[str]:
        return ["error_text", "detection_mode", "k8s_events"]

    def fetch_event(self, event_id: str, **kwargs: Any) -> RunEvent:
        """
        Fetches a pod's failure details.
        
        Args:
            event_id: The pod name.
            **kwargs: Must contain 'namespace'. Defaults to 'default'.
        """
        namespace = kwargs.get("namespace", "default")
        return self.from_api(namespace, event_id)

    def from_api(self, namespace: str, pod_name: str) -> RunEvent:
        """
        Actively queries the Kubernetes API for the given Pod.
        """
        try:
            from kubernetes import client, config
        except ImportError:
            raise MissingDependenciesError("The 'kubernetes' package is required to use KubernetesAdapter.")
            
        try:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
                
            v1 = client.CoreV1Api()
            pod = v1.read_namespaced_pod(name=pod_name, namespace=namespace)
            
            try:
                logs = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace, tail_lines=50)
            except Exception:
                logs = "Logs not available"
                
            events_list = v1.list_namespaced_event(namespace=namespace, field_selector=f"involvedObject.name={pod_name}")
            warning_events = [e.message for e in events_list.items if e.type == "Warning"]
            
            payload = {
                "namespace": namespace,
                "pod_name": pod_name,
                "phase": pod.status.phase,
                "reason": pod.status.container_statuses[0].state.waiting.reason if pod.status.container_statuses and pod.status.container_statuses[0].state.waiting else "Unknown",
                "logs": logs,
                "events": warning_events,
                "labels": pod.metadata.labels or {},
                "image": pod.spec.containers[0].image if pod.spec.containers else "unknown"
            }
            return self.parse_raw(payload)
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch K8s data: {e}")

    def parse_raw(self, payload: Dict[str, Any]) -> RunEvent:
        namespace = payload.get("namespace", "default")
        pod_name = payload.get("pod_name", "unknown")
        repo = f"{namespace}/{pod_name}"
        
        reason = payload.get("reason", "Error")
        logs = payload.get("logs", "")
        events = payload.get("events", [])
        labels = payload.get("labels", {})

        commit_sha = labels.get("commit_sha") or labels.get("app.kubernetes.io/revision")
        
        error_text_combined = f"REASON: {reason}\\nLOGS:\\n{logs}\\nEVENTS:\\n" + "\\n".join(events)
        
        failed_step = FailedStepInfo(
            name=f"Pod {pod_name} ({reason})",
            error_text=error_text_combined,
            metadata={
                "job_file": pod_name,
                "runner_image": payload.get("image", "unknown"),
                "step_type": "container",
                "detection_mode": "per_step_error"
            }
        )

        return RunEvent(
            source=self.source_name,
            run_id=f"k8s-{pod_name}-{datetime.utcnow().timestamp()}",
            status="failure",
            failed_steps=[failed_step],
            available_signals=self.available_signals,
            metadata={
                "repo": repo,
                "workflow": "kubernetes-deployment",
                "commit_sha": commit_sha,
                "k8s_events": events,
                "k8s_reason": reason,
                "detection_mode": "per_step_error"
            }
        )
