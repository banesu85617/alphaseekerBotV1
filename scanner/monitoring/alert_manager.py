"""
警报管理器
实现实时市场监控和警报系统
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """警报严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """警报状态"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    EXPIRED = "expired"


@dataclass
class Alert:
    """警报"""
    id: str
    type: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    source: str = "scanner"
    status: AlertStatus = AlertStatus.ACTIVE
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'type': self.type,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'status': self.status.value,
            'data': self.data,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """从字典创建警报"""
        return cls(
            id=data['id'],
            type=data['type'],
            severity=AlertSeverity(data['severity']),
            title=data['title'],
            message=data['message'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=data.get('source', 'scanner'),
            status=AlertStatus(data.get('status', 'active')),
            data=data.get('data', {}),
            metadata=data.get('metadata', {})
        )


@dataclass
class AlertRule:
    """警报规则"""
    id: str
    name: str
    description: str
    alert_type: str
    severity: AlertSeverity
    condition: Callable[[Any], bool]
    cooldown: timedelta = timedelta(minutes=5)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def check_condition(self, data: Any) -> bool:
        """检查条件"""
        try:
            return self.condition(data)
        except Exception as e:
            logger.error(f"Error checking alert condition for {self.id}: {e}")
            return False


class AlertHandler(ABC):
    """警报处理器基类"""
    
    @abstractmethod
    async def handle_alert(self, alert: Alert) -> bool:
        """处理警报"""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """启动处理器"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """停止处理器"""
        pass


class LoggingAlertHandler(AlertHandler):
    """日志警报处理器"""
    
    def __init__(self):
        self.enabled = True
    
    async def handle_alert(self, alert: Alert) -> bool:
        """处理警报"""
        if not self.enabled:
            return False
        
        log_message = f"[{alert.severity.value.upper()}] {alert.title}: {alert.message}"
        
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(log_message)
        elif alert.severity == AlertSeverity.ERROR:
            logger.error(log_message)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        return True
    
    async def start(self) -> None:
        """启动处理器"""
        self.enabled = True
        logger.info("LoggingAlertHandler started")
    
    async def stop(self) -> None:
        """停止处理器"""
        self.enabled = False
        logger.info("LoggingAlertHandler stopped")


class WebhookAlertHandler(AlertHandler):
    """Webhook警报处理器"""
    
    def __init__(self, webhook_url: str, timeout: float = 10.0):
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.enabled = False
        self.session = None
    
    async def handle_alert(self, alert: Alert) -> bool:
        """处理警报"""
        if not self.enabled or not self.webhook_url:
            return False
        
        try:
            import aiohttp
            
            payload = {
                'alert': alert.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    success = response.status < 400
                    if success:
                        logger.info(f"Alert webhook sent successfully for alert {alert.id}")
                    else:
                        logger.error(f"Alert webhook failed with status {response.status}")
                    return success
                    
        except Exception as e:
            logger.error(f"Error sending alert webhook: {e}")
            return False
    
    async def start(self) -> None:
        """启动处理器"""
        self.enabled = True
        logger.info(f"WebhookAlertHandler started for {self.webhook_url}")
    
    async def stop(self) -> None:
        """停止处理器"""
        self.enabled = False
        if self.session:
            await self.session.close()
        logger.info("WebhookAlertHandler stopped")


class EmailAlertHandler(AlertHandler):
    """邮件警报处理器"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, 
                 from_email: str, to_emails: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.enabled = False
    
    async def handle_alert(self, alert: Alert) -> bool:
        """处理警报"""
        if not self.enabled:
            return False
        
        try:
            import aiosmtplib
            from email.message import EmailMessage
            
            subject = f"[{alert.severity.value.upper()}] {alert.title}"
            body = f"""
扫描警报通知

警报ID: {alert.id}
类型: {alert.type}
严重程度: {alert.severity.value}
时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
来源: {alert.source}

消息:
{alert.message}

数据:
{json.dumps(alert.data, indent=2, default=str)}

元数据:
{json.dumps(alert.metadata, indent=2, default=str)}
            """
            
            msg = EmailMessage()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = subject
            msg.set_content(body)
            
            await aiosmtplib.send(
                msg,
                hostname=self.smtp_server,
                port=self.smtp_port,
                username=self.username,
                password=self.password,
                use_tls=True
            )
            
            logger.info(f"Alert email sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending alert email: {e}")
            return False
    
    async def start(self) -> None:
        """启动处理器"""
        self.enabled = True
        logger.info("EmailAlertHandler started")
    
    async def stop(self) -> None:
        """停止处理器"""
        self.enabled = False
        logger.info("EmailAlertHandler stopped")


class AlertManager:
    """警报管理器"""
    
    def __init__(self, max_alerts: int = 1000):
        """
        初始化警报管理器
        
        Args:
            max_alerts: 最大警报数量
        """
        self.max_alerts = max_alerts
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        self.handlers: List[AlertHandler] = []
        self.active_alerts: Dict[str, Alert] = {}
        
        # 警报统计
        self.stats = {
            'total_alerts': 0,
            'active_alerts': 0,
            'resolved_alerts': 0,
            'alerts_by_severity': {severity.value: 0 for severity in AlertSeverity},
            'alerts_by_type': {}
        }
        
        # 默认处理器
        self._setup_default_handlers()
        
        # 默认规则
        self._setup_default_rules()
        
        logger.info("AlertManager initialized")
    
    def _setup_default_handlers(self):
        """设置默认处理器"""
        # 添加日志处理器
        self.add_handler(LoggingAlertHandler())
    
    def _setup_default_rules(self):
        """设置默认规则"""
        # 高价值机会警报
        high_opportunity_rule = AlertRule(
            id="high_opportunity",
            name="High Value Opportunity",
            description="High-scored trading opportunities detected",
            alert_type="opportunity",
            severity=AlertSeverity.INFO,
            condition=lambda data: data.get('score', 0) > 0.9
        )
        self.add_rule(high_opportunity_rule)
        
        # 系统错误警报
        error_rule = AlertRule(
            id="system_error",
            name="System Error",
            description="System errors detected",
            alert_type="system",
            severity=AlertSeverity.ERROR,
            condition=lambda data: data.get('error_count', 0) > 5
        )
        self.add_rule(error_rule)
        
        # 性能警报
        performance_rule = AlertRule(
            id="performance_issue",
            name="Performance Issue",
            description="Performance degradation detected",
            alert_type="performance",
            severity=AlertSeverity.WARNING,
            condition=lambda data: data.get('duration', 0) > 30.0
        )
        self.add_rule(performance_rule)
    
    def add_handler(self, handler: AlertHandler):
        """添加警报处理器"""
        self.handlers.append(handler)
        logger.info(f"Added alert handler: {handler.__class__.__name__}")
    
    def add_rule(self, rule: AlertRule):
        """添加警报规则"""
        self.alert_rules[rule.id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    async def start_handlers(self):
        """启动所有处理器"""
        for handler in self.handlers:
            try:
                await handler.start()
            except Exception as e:
                logger.error(f"Error starting handler {handler.__class__.__name__}: {e}")
    
    async def stop_handlers(self):
        """停止所有处理器"""
        for handler in self.handlers:
            try:
                await handler.stop()
            except Exception as e:
                logger.error(f"Error stopping handler {handler.__class__.__name__}: {e}")
    
    async def send_alert(self, alert_data: Dict[str, Any]) -> bool:
        """
        发送警报
        
        Args:
            alert_data: 警报数据
            
        Returns:
            True如果发送成功
        """
        try:
            # 创建警报
            alert = Alert(
                id=alert_data.get('id', f"alert_{len(self.alerts) + 1}_{int(datetime.now().timestamp())}"),
                type=alert_data.get('type', 'system'),
                severity=AlertSeverity(alert_data.get('severity', 'info')),
                title=alert_data.get('title', 'Scanner Alert'),
                message=alert_data.get('message', 'No message provided'),
                timestamp=datetime.now(),
                source=alert_data.get('source', 'scanner'),
                data=alert_data.get('data', {}),
                metadata=alert_data.get('metadata', {})
            )
            
            # 检查冷却时间
            if not self._check_cooldown(alert):
                return False
            
            # 添加到列表
            self.alerts.append(alert)
            self.active_alerts[alert.id] = alert
            
            # 更新统计
            self._update_stats(alert)
            
            # 限制列表大小
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]
            
            # 分发给处理器
            success_count = 0
            for handler in self.handlers:
                try:
                    if await handler.handle_alert(alert):
                        success_count += 1
                except Exception as e:
                    logger.error(f"Error in alert handler {handler.__class__.__name__}: {e}")
            
            logger.info(f"Alert {alert.id} sent to {success_count} handlers")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False
    
    async def check_rules(self, data: Any) -> List[Alert]:
        """检查警报规则"""
        triggered_alerts = []
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            try:
                if rule.check_condition(data):
                    alert_data = {
                        'type': rule.alert_type,
                        'severity': rule.severity.value,
                        'title': f"Rule Triggered: {rule.name}",
                        'message': rule.description,
                        'data': {'trigger_data': data, 'rule_id': rule.id},
                        'metadata': {'rule': rule.id}
                    }
                    
                    alert_id = f"rule_{rule.id}_{int(datetime.now().timestamp())}"
                    alert_data['id'] = alert_id
                    
                    if await self.send_alert(alert_data):
                        triggered_alerts.append(self.active_alerts.get(alert_id))
                        
            except Exception as e:
                logger.error(f"Error checking rule {rule.id}: {e}")
        
        return triggered_alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认警报"""
        alert = self.active_alerts.get(alert_id)
        if alert and alert.status == AlertStatus.ACTIVE:
            alert.status = AlertStatus.ACKNOWLEDGED
            logger.info(f"Alert {alert_id} acknowledged")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决警报"""
        alert = self.active_alerts.get(alert_id)
        if alert:
            alert.status = AlertStatus.RESOLVED
            del self.active_alerts[alert_id]
            logger.info(f"Alert {alert_id} resolved")
            return True
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """获取活跃警报"""
        active = [alert for alert in self.alerts if alert.status == AlertStatus.ACTIVE]
        
        if severity:
            active = [alert for alert in active if alert.severity == severity]
        
        return active
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取警报历史"""
        return self.alerts[-limit:] if limit > 0 else self.alerts
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(self.active_alerts),
            'resolved_alerts': len([a for a in self.alerts if a.status == AlertStatus.RESOLVED]),
            'alerts_by_severity': self.stats['alerts_by_severity'],
            'alerts_by_type': self.stats['alerts_by_type'],
            'rules_count': len(self.alert_rules),
            'handlers_count': len(self.handlers),
            'recent_alerts': [alert.to_dict() for alert in self.alerts[-10:]]
        }
    
    def _check_cooldown(self, alert: Alert) -> bool:
        """检查冷却时间"""
        # 查找同类型的最近警报
        recent_alerts = [
            a for a in self.alerts 
            if a.type == alert.type and 
            (datetime.now() - a.timestamp).total_seconds() < 300  # 5分钟冷却
        ]
        
        return len(recent_alerts) == 0  # 如果没有最近的同类型警报，允许发送
    
    def _update_stats(self, alert: Alert):
        """更新统计信息"""
        self.stats['total_alerts'] += 1
        self.stats['alerts_by_severity'][alert.severity.value] += 1
        self.stats['alerts_by_type'][alert.type] = self.stats['alerts_by_type'].get(alert.type, 0) + 1
        
        if alert.status == AlertStatus.ACTIVE:
            self.stats['active_alerts'] += 1
        elif alert.status == AlertStatus.RESOLVED:
            self.stats['resolved_alerts'] += 1
    
    def cleanup_expired_alerts(self, max_age: timedelta = timedelta(hours=24)):
        """清理过期警报"""
        cutoff_time = datetime.now() - max_age
        
        # 清理已解决的过期警报
        expired = [
            alert for alert in self.alerts 
            if alert.status in [AlertStatus.RESOLVED, AlertStatus.EXPIRED] and 
            alert.timestamp < cutoff_time
        ]
        
        for alert in expired:
            self.alerts.remove(alert)
        
        logger.info(f"Cleaned up {len(expired)} expired alerts")
        return len(expired)
    
    def enable_rule(self, rule_id: str) -> bool:
        """启用规则"""
        rule = self.alert_rules.get(rule_id)
        if rule:
            rule.enabled = True
            logger.info(f"Enabled alert rule: {rule_id}")
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """禁用规则"""
        rule = self.alert_rules.get(rule_id)
        if rule:
            rule.enabled = False
            logger.info(f"Disabled alert rule: {rule_id}")
            return True
        return False


# 便利函数
def create_webhook_alert(alert_type: str, title: str, message: str, 
                        severity: str = "info", data: Dict[str, Any] = None) -> Dict[str, Any]:
    """创建Webhook警报数据"""
    return {
        'type': alert_type,
        'severity': severity,
        'title': title,
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'data': data or {}
    }


def create_opportunity_alert(symbol: str, score: float, reason: str) -> Dict[str, Any]:
    """创建机会警报"""
    severity = "critical" if score > 0.9 else "warning" if score > 0.8 else "info"
    
    return create_webhook_alert(
        alert_type="opportunity",
        title=f"Trading Opportunity: {symbol}",
        message=f"High-scored opportunity detected: {reason}",
        severity=severity,
        data={
            'symbol': symbol,
            'score': score,
            'reason': reason
        }
    )


def create_performance_alert(metric_name: str, value: float, threshold: float) -> Dict[str, Any]:
    """创建性能警报"""
    severity = "critical" if value > threshold * 1.5 else "warning"
    
    return create_webhook_alert(
        alert_type="performance",
        title=f"Performance Issue: {metric_name}",
        message=f"{metric_name} value {value:.2f} exceeds threshold {threshold:.2f}",
        severity=severity,
        data={
            'metric': metric_name,
            'value': value,
            'threshold': threshold
        }
    )


# 示例使用
if __name__ == "__main__":
    async def test_alert_manager():
        # 创建警报管理器
        alert_manager = AlertManager()
        
        # 添加Webhook处理器（示例）
        webhook_handler = WebhookAlertHandler("https://hooks.slack.com/test")
        alert_manager.add_handler(webhook_handler)
        
        # 启动处理器
        await alert_manager.start_handlers()
        
        # 发送测试警报
        test_alert = create_opportunity_alert("BTCUSDT", 0.95, "Strong technical signals")
        await alert_manager.send_alert(test_alert)
        
        # 检查规则
        rule_data = {'score': 0.92, 'symbol': 'ETHUSDT'}
        triggered = await alert_manager.check_rules(rule_data)
        print(f"Triggered {len(triggered)} alerts from rules")
        
        # 获取统计信息
        stats = alert_manager.get_statistics()
        print(f"Alert statistics: {json.dumps(stats, indent=2, default=str)}")
        
        # 停止处理器
        await alert_manager.stop_handlers()
    
    # 运行测试
    import asyncio
    asyncio.run(test_alert_manager())