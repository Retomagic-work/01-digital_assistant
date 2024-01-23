import {forwardRef, Inject, Injectable, Logger} from "@nestjs/common";
import {Connection} from "rabbitmq-client";
import {RequestsDto} from "../requests/dto/requests.dto";
import {RequestsService} from "../requests/requests.service";
import {RabbitmqRequestsDto} from "./dto/rabbitmq-requests.dto";
import {DocumentsDto} from "../documents/dto/documents.dto";

@Injectable()
export class RabbitmqService {
    private readonly logger = new Logger(RabbitmqService.name);
    private rabbit = new Connection(`amqp://${process.env.RABBITMQ_DEFAULT_USER}:${process.env.RABBITMQ_DEFAULT_PASS}@rabbitmq:5672`)

    constructor(
        @Inject(forwardRef(() => RequestsService)) private requestsService: RequestsService,
    ) {
    }

    private async onModuleInit() {
        this.rabbit.on('error', (err) => {
            this.logger.log('RabbitMQ connection error', err)
        })

        this.rabbit.on('connection', () => {
            this.logger.log('Connection successfully (re)established')
        })

        this.rabbit.createConsumer({
            queue: 'responses',
            queueOptions: {durable: true},
            qos: {prefetchCount: 1},
        }, async (msg) => {
            console.log("1", typeof msg.body, msg.body)
            const body: RabbitmqRequestsDto = JSON.parse(msg.body)

            console.log("2", body)

            await this.requestsService.updateStatus(body)

            console.log("3", body)

            this.logger.log('received message (responses)', JSON.parse(msg.body))
        })
    }

    async sendMessage(data: RequestsDto) {
        const pub = this.rabbit.createPublisher({
            confirm: true,
            maxAttempts: 100
        })

        await pub.send('requests', data)
    }

    async sendMessageUpload(data: DocumentsDto) {
        const pub = this.rabbit.createPublisher({
            confirm: true,
            maxAttempts: 100
        })

        await pub.send('upload', data)
    }
}
