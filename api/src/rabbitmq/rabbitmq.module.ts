import {forwardRef, Module} from "@nestjs/common";
import {RabbitmqService} from "./rabbitmq.service";
import {RequestsModule} from "../requests/requests.module";

@Module({
    imports: [forwardRef(() => RequestsModule)],
    controllers: [],
    providers: [RabbitmqService],
    exports: [RabbitmqService],
})
export class RabbitmqModule {
}
